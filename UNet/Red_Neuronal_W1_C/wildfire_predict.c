/*
 * wildfire_predict.c
 *
 * Predicción de incendios forestales con UNet2D en C puro.
 * Carga los pesos exportados por export_weights.py y reproduce
 * exactamente el forward pass de PyTorch.
 *
 * Modos de uso:
 *   1) Validación numérica contra PyTorch (no necesita GDAL):
 *        ./wildfire_predict --test weights_bin/
 *
 *   2) Predicción sobre imagen TIFF (requiere GDAL):
 *        ./wildfire_predict --weights weights_bin/ \
 *                           --image   imagen.tif  \
 *                           --output  pred.bin    \
 *                           --size    128
 *
 * Compilar sin GDAL (solo modo --test):
 *   gcc -O2 -o wildfire_predict wildfire_predict.c -lm
 *
 * Compilar con GDAL (modo --image):
 *   gcc -O2 -DWITH_GDAL -o wildfire_predict wildfire_predict.c \
 *       $(gdal-config --cflags) $(gdal-config --libs) -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef WITH_GDAL
#include "gdal.h"
#include "cpl_conv.h"
#endif


/* ================================================================
 * SECCIÓN 1 — TENSOR 4D
 * ================================================================ */
typedef struct {
    float *data;
    int batch, channels, height, width;
} Tensor4D;

static Tensor4D* tensor_create(int b, int c, int h, int w) {
    Tensor4D *t = (Tensor4D*)malloc(sizeof(Tensor4D));
    t->batch = b; t->channels = c; t->height = h; t->width = w;
    t->data  = (float*)calloc((size_t)b*c*h*w, sizeof(float));
    return t;
}

static inline float tget(const Tensor4D *t, int b, int c, int h, int w) {
    return t->data[b*(t->channels*t->height*t->width)
                  + c*(t->height*t->width)
                  + h*t->width + w];
}

static inline void tset(Tensor4D *t, int b, int c, int h, int w, float v) {
    t->data[b*(t->channels*t->height*t->width)
           + c*(t->height*t->width)
           + h*t->width + w] = v;
}

static void tensor_free(Tensor4D *t) {
    if (t) { free(t->data); free(t); }
}


/* ================================================================
 * SECCIÓN 2 — CARGA DE PESOS DESDE ARCHIVO BINARIO
 * ================================================================ */

/*
 * load_bin
 * Lee exactamente n floats de un archivo binario.
 * Termina el programa si el archivo no existe o tiene tamaño incorrecto.
 */
static float* load_bin(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Error] No se pudo abrir: %s\n", path);
        exit(1);
    }
    float *buf = (float*)malloc((size_t)n * sizeof(float));
    size_t read = fread(buf, sizeof(float), n, f);
    fclose(f);
    if ((int)read != n) {
        fprintf(stderr, "[Error] %s: esperaba %d floats, leyó %zu\n",
                path, n, read);
        exit(1);
    }
    return buf;
}

/*
 * path_join
 * Concatena dir + "/" + filename en un buffer estático.
 * Solo para llamadas donde el resultado se usa antes de la siguiente llamada.
 */
static char g_path_buf[1024];
static const char* pjoin(const char *dir, const char *fname) {
    snprintf(g_path_buf, sizeof(g_path_buf), "%s/%s", dir, fname);
    return g_path_buf;
}


/* ================================================================
 * SECCIÓN 3 — OPERACIONES DE RED NEURONAL
 * ================================================================ */

/* Conv2d 3×3, padding=1, stride=1 */
static void conv2d_3x3(const Tensor4D *in, const float *W, const float *b,
                       Tensor4D *out) {
    const int B = in->batch, Ci = in->channels, H = in->height, W_ = in->width;
    const int Co = out->channels;
    for (int bn = 0; bn < B;  bn++)
    for (int co = 0; co < Co; co++)
    for (int oh = 0; oh < H;  oh++)
    for (int ow = 0; ow < W_; ow++) {
        float s = b ? b[co] : 0.0f;
        for (int ci = 0; ci < Ci; ci++)
        for (int ky = 0; ky < 3;  ky++)
        for (int kx = 0; kx < 3;  kx++) {
            int ih = oh - 1 + ky, iw = ow - 1 + kx;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W_)
                s += tget(in, bn, ci, ih, iw)
                   * W[co*(Ci*9) + ci*9 + ky*3 + kx];
        }
        tset(out, bn, co, oh, ow, s);
    }
}

/* Conv2d 1×1 (capa de salida) */
static void conv2d_1x1(const Tensor4D *in, const float *W, const float *b,
                       Tensor4D *out) {
    const int B = in->batch, Ci = in->channels, H = in->height, W_ = in->width;
    const int Co = out->channels;
    for (int bn = 0; bn < B;  bn++)
    for (int co = 0; co < Co; co++)
    for (int oh = 0; oh < H;  oh++)
    for (int ow = 0; ow < W_; ow++) {
        float s = b ? b[co] : 0.0f;
        for (int ci = 0; ci < Ci; ci++)
            s += tget(in, bn, ci, oh, ow) * W[co*Ci + ci];
        tset(out, bn, co, oh, ow, s);
    }
}

/* BatchNorm2d — modo inferencia */
static void batch_norm(Tensor4D *t, const float *gamma, const float *beta,
                       const float *mean, const float *var, float eps) {
    for (int b = 0; b < t->batch;    b++)
    for (int c = 0; c < t->channels; c++) {
        float inv = 1.0f / sqrtf(var[c] + eps);
        for (int h = 0; h < t->height; h++)
        for (int w = 0; w < t->width;  w++) {
            float x = tget(t, b, c, h, w);
            tset(t, b, c, h, w, gamma[c]*(x - mean[c])*inv + beta[c]);
        }
    }
}

/* ReLU in-place */
static void relu(Tensor4D *t) {
    int n = t->batch * t->channels * t->height * t->width;
    for (int i = 0; i < n; i++)
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
}

/* MaxPool2d 2×2, stride=2 */
static Tensor4D* maxpool2(const Tensor4D *in) {
    Tensor4D *out = tensor_create(in->batch, in->channels,
                                  in->height/2, in->width/2);
    for (int b = 0; b < in->batch;    b++)
    for (int c = 0; c < in->channels; c++)
    for (int h = 0; h < out->height;  h++)
    for (int w = 0; w < out->width;   w++) {
        float mx = tget(in, b, c, h*2,   w*2);
        float v1 = tget(in, b, c, h*2,   w*2+1);
        float v2 = tget(in, b, c, h*2+1, w*2);
        float v3 = tget(in, b, c, h*2+1, w*2+1);
        if (v1 > mx) mx = v1;
        if (v2 > mx) mx = v2;
        if (v3 > mx) mx = v3;
        tset(out, b, c, h, w, mx);
    }
    return out;
}

/*
 * ConvTranspose2d 2×2, stride=2 — SIN ReLU
 * (CORRECCIÓN 2: la activación solo está en double_conv)
 *
 * Pesos en formato PyTorch: [C_in, C_out, 2, 2]
 */
static Tensor4D* conv_transpose2(const Tensor4D *in, const float *W,
                                  const float *b, int C_out) {
    const int B = in->batch, Ci = in->channels;
    const int Hi = in->height, Wi = in->width;
    Tensor4D *out = tensor_create(B, C_out, Hi*2, Wi*2);

    /* Pre-cargar bias */
    if (b) {
        for (int bn = 0; bn < B;    bn++)
        for (int co = 0; co < C_out; co++)
        for (int oh = 0; oh < Hi*2;  oh++)
        for (int ow = 0; ow < Wi*2;  ow++)
            tset(out, bn, co, oh, ow, b[co]);
    }

    for (int bn = 0; bn < B;  bn++)
    for (int ci = 0; ci < Ci; ci++)
    for (int ih = 0; ih < Hi; ih++)
    for (int iw = 0; iw < Wi; iw++) {
        float v = tget(in, bn, ci, ih, iw);
        for (int co = 0; co < C_out; co++)
        for (int ky = 0; ky < 2;     ky++)
        for (int kx = 0; kx < 2;     kx++) {
            int w_idx = ci*(C_out*4) + co*4 + ky*2 + kx;
            float prev = tget(out, bn, co, ih*2+ky, iw*2+kx);
            tset(out, bn, co, ih*2+ky, iw*2+kx, prev + v*W[w_idx]);
        }
    }
    return out;
}

/*
 * concat_channels — CORRECCIÓN 3: encoder primero
 * Equivale a torch.cat([enc, dec], dim=1)
 */
static Tensor4D* concat(const Tensor4D *enc, const Tensor4D *dec) {
    int C_tot = enc->channels + dec->channels;
    Tensor4D *out = tensor_create(enc->batch, C_tot,
                                  enc->height, enc->width);
    int sp = enc->height * enc->width;
    for (int b = 0; b < enc->batch; b++) {
        for (int c = 0; c < enc->channels; c++) {
            float *src = enc->data + b*(enc->channels*sp) + c*sp;
            float *dst = out->data + b*(C_tot*sp) + c*sp;
            memcpy(dst, src, sp * sizeof(float));
        }
        for (int c = 0; c < dec->channels; c++) {
            float *src = dec->data + b*(dec->channels*sp) + c*sp;
            float *dst = out->data + b*(C_tot*sp) + (enc->channels+c)*sp;
            memcpy(dst, src, sp * sizeof(float));
        }
    }
    return out;
}

/* Softmax sobre el eje de canales (dim=1) */
static void softmax_channels(Tensor4D *t) {
    const int B = t->batch, C = t->channels;
    const int H = t->height, W = t->width;
    for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        /* Estabilidad numérica: restar el máximo antes de exp */
        float mx = tget(t, b, 0, h, w);
        for (int c = 1; c < C; c++) {
            float v = tget(t, b, c, h, w);
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            float e = expf(tget(t, b, c, h, w) - mx);
            tset(t, b, c, h, w, e);
            sum += e;
        }
        for (int c = 0; c < C; c++)
            tset(t, b, c, h, w, tget(t, b, c, h, w) / sum);
    }
}


/* ================================================================
 * SECCIÓN 4 — BLOQUE DOUBLE CONV
 *
 * Estructura de parámetros para un bloque DoubleConv2D.
 * Todos los punteros apuntan a arreglos float32 cargados con load_bin.
 * ================================================================ */
typedef struct {
    const float *w1, *b1, *g1, *beta1, *m1, *v1;  /* conv1 + bn1 */
    const float *w2, *b2, *g2, *beta2, *m2, *v2;  /* conv2 + bn2 */
    int c_in, c_out;
} DCParams;

static void free_dc(DCParams *p) {
    free((void*)p->w1); free((void*)p->b1);
    free((void*)p->g1); free((void*)p->beta1);
    free((void*)p->m1); free((void*)p->v1);
    free((void*)p->w2); free((void*)p->b2);
    free((void*)p->g2); free((void*)p->beta2);
    free((void*)p->m2); free((void*)p->v2);
}

static DCParams load_dc(const char *dir, const char *pfx,
                        int c_in, int c_out) {
    DCParams p;
    p.c_in  = c_in;
    p.c_out = c_out;

    /* Nombre de archivo: "{dir}/{pfx}_conv1_w.bin" etc. */
    char name[256];

#define LBIN(field, suffix, n) \
    snprintf(name, sizeof(name), "%s_%s.bin", pfx, suffix); \
    p.field = load_bin(pjoin(dir, name), n)

    LBIN(w1,    "conv1_w",    c_out * c_in  * 9);
    LBIN(b1,    "conv1_b",    c_out);
    LBIN(g1,    "bn1_gamma",  c_out);
    LBIN(beta1, "bn1_beta",   c_out);
    LBIN(m1,    "bn1_mean",   c_out);
    LBIN(v1,    "bn1_var",    c_out);
    LBIN(w2,    "conv2_w",    c_out * c_out * 9);
    LBIN(b2,    "conv2_b",    c_out);
    LBIN(g2,    "bn2_gamma",  c_out);
    LBIN(beta2, "bn2_beta",   c_out);
    LBIN(m2,    "bn2_mean",   c_out);
    LBIN(v2,    "bn2_var",    c_out);
#undef LBIN

    return p;
}

/*
 * apply_dc — aplica un bloque DoubleConv sobre 'in' y devuelve el resultado.
 * Libera 'in' internamente para simplificar el grafo de memoria en el forward.
 * Si free_input=0, no libera (útil para skip connections).
 */
static Tensor4D* apply_dc(const Tensor4D *in, const DCParams *p,
                           int free_input) {
    Tensor4D *mid = tensor_create(in->batch, p->c_out,
                                  in->height, in->width);
    conv2d_3x3(in, p->w1, p->b1, mid);
    batch_norm(mid, p->g1, p->beta1, p->m1, p->v1, 1e-5f);
    relu(mid);

    Tensor4D *out = tensor_create(in->batch, p->c_out,
                                  in->height, in->width);
    conv2d_3x3(mid, p->w2, p->b2, out);
    batch_norm(out, p->g2, p->beta2, p->m2, p->v2, 1e-5f);
    relu(out);

    tensor_free(mid);
    if (free_input) tensor_free((Tensor4D*)in);
    return out;
}


/* ================================================================
 * SECCIÓN 5 — PARÁMETROS COMPLETOS DEL MODELO
 * ================================================================ */
typedef struct {
    /* Encoder */
    DCParams enc1, enc2, enc3, enc4, bottleneck;

    /* Decoder — pesos de ConvTranspose */
    float *up4_w, *up4_b;
    float *up3_w, *up3_b;
    float *up2_w, *up2_b;
    float *up1_w, *up1_b;

    /* Decoder — bloques DoubleConv */
    DCParams dec4, dec3, dec2, dec1;

    /* Capa de salida Conv1×1 */
    float *out_w, *out_b;
} UNetParams;

static UNetParams load_unet_params(const char *dir) {
    UNetParams n;

    /* Encoder */
    n.enc1       = load_dc(dir, "enc1",        4,   64);
    n.enc2       = load_dc(dir, "enc2",        64, 128);
    n.enc3       = load_dc(dir, "enc3",       128, 256);
    n.enc4       = load_dc(dir, "enc4",       256, 512);
    n.bottleneck = load_dc(dir, "bottleneck", 512,1024);

    /* ConvTranspose 2×2 — formato [C_in, C_out, 2, 2] */
    n.up4_w = load_bin(pjoin(dir,"upconv4_w.bin"), 1024*512*4);
    n.up4_b = load_bin(pjoin(dir,"upconv4_b.bin"), 512);
    n.up3_w = load_bin(pjoin(dir,"upconv3_w.bin"), 512*256*4);
    n.up3_b = load_bin(pjoin(dir,"upconv3_b.bin"), 256);
    n.up2_w = load_bin(pjoin(dir,"upconv2_w.bin"), 256*128*4);
    n.up2_b = load_bin(pjoin(dir,"upconv2_b.bin"), 128);
    n.up1_w = load_bin(pjoin(dir,"upconv1_w.bin"), 128*64*4);
    n.up1_b = load_bin(pjoin(dir,"upconv1_b.bin"), 64);

    /* Decoder DoubleConv — c_in = c_up + c_skip */
    n.dec4 = load_dc(dir, "dec4", 1024, 512);
    n.dec3 = load_dc(dir, "dec3",  512, 256);
    n.dec2 = load_dc(dir, "dec2",  256, 128);
    n.dec1 = load_dc(dir, "dec1",  128,  64);

    /* Capa de salida */
    n.out_w = load_bin(pjoin(dir,"out_conv_w.bin"), 64*2);
    n.out_b = load_bin(pjoin(dir,"out_conv_b.bin"), 2);

    return n;
}

static void free_unet_params(UNetParams *n) {
    free_dc(&n->enc1); free_dc(&n->enc2);
    free_dc(&n->enc3); free_dc(&n->enc4);
    free_dc(&n->bottleneck);
    free(n->up4_w); free(n->up4_b);
    free(n->up3_w); free(n->up3_b);
    free(n->up2_w); free(n->up2_b);
    free(n->up1_w); free(n->up1_b);
    free_dc(&n->dec4); free_dc(&n->dec3);
    free_dc(&n->dec2); free_dc(&n->dec1);
    free(n->out_w);  free(n->out_b);
}


/* ================================================================
 * SECCIÓN 6 — FORWARD PASS COMPLETO
 * ================================================================ */
static Tensor4D* unet_forward(const Tensor4D *input, const UNetParams *n) {

    /* ---- Encoder ---- */
    Tensor4D *e1 = apply_dc(input, &n->enc1, 0);   /* [1, 64,  H,   W  ] */

    Tensor4D *p1 = maxpool2(e1);
    Tensor4D *e2 = apply_dc(p1,   &n->enc2, 1);    /* [1,128,  H/2, W/2] */

    Tensor4D *p2 = maxpool2(e2);
    Tensor4D *e3 = apply_dc(p2,   &n->enc3, 1);    /* [1,256,  H/4, W/4] */

    Tensor4D *p3 = maxpool2(e3);
    Tensor4D *e4 = apply_dc(p3,   &n->enc4, 1);    /* [1,512,  H/8, W/8] */

    Tensor4D *p4 = maxpool2(e4);

    /* ---- Bottleneck ---- */
    Tensor4D *bn = apply_dc(p4, &n->bottleneck, 1); /* [1,1024,H/16,W/16] */

    /* ---- Decoder ---- */
    /* d4 */
    Tensor4D *up4  = conv_transpose2(bn,  n->up4_w, n->up4_b, 512);
    tensor_free(bn);
    Tensor4D *cat4 = concat(e4, up4);           /* encoder primero: CORRECCIÓN 3 */
    tensor_free(e4); tensor_free(up4);
    Tensor4D *d4   = apply_dc(cat4, &n->dec4, 1);

    /* d3 */
    Tensor4D *up3  = conv_transpose2(d4,  n->up3_w, n->up3_b, 256);
    tensor_free(d4);
    Tensor4D *cat3 = concat(e3, up3);
    tensor_free(e3); tensor_free(up3);
    Tensor4D *d3   = apply_dc(cat3, &n->dec3, 1);

    /* d2 */
    Tensor4D *up2  = conv_transpose2(d3,  n->up2_w, n->up2_b, 128);
    tensor_free(d3);
    Tensor4D *cat2 = concat(e2, up2);
    tensor_free(e2); tensor_free(up2);
    Tensor4D *d2   = apply_dc(cat2, &n->dec2, 1);

    /* d1 */
    Tensor4D *up1  = conv_transpose2(d2,  n->up1_w, n->up1_b, 64);
    tensor_free(d2);
    Tensor4D *cat1 = concat(e1, up1);
    tensor_free(e1); tensor_free(up1);
    Tensor4D *d1   = apply_dc(cat1, &n->dec1, 1);

    /* ---- Salida Conv1×1 ---- */
    Tensor4D *out = tensor_create(d1->batch, 2, d1->height, d1->width);
    conv2d_1x1(d1, n->out_w, n->out_b, out);
    tensor_free(d1);

    return out;   /* logits [1, 2, H, W] */
}


/* ================================================================
 * SECCIÓN 7 — CARGA DE IMAGEN
 * ================================================================ */

/* Carga imagen desde archivo .bin de prueba generado por export_weights.py */
static Tensor4D* load_image_bin(const char *path, int B, int C, int H, int W) {
    printf("Cargando imagen desde binario: %s\n", path);
    Tensor4D *t = tensor_create(B, C, H, W);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr,"No se pudo abrir %s\n", path); exit(1); }
    size_t nr = fread(t->data, sizeof(float), (size_t)B*C*H*W, f); (void)nr;
    fclose(f);
    return t;
}

#ifdef WITH_GDAL
/* Carga imagen TIFF de 4 bandas UInt16, normaliza dividiendo por 12500 */
static Tensor4D* load_image_tiff(const char *path, int patch_size) {
    printf("Cargando imagen TIFF: %s\n", path);
    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(path, GA_ReadOnly);
    if (!ds) { fprintf(stderr,"Error abriendo TIFF: %s\n", path); exit(1); }

    int bands  = GDALGetRasterCount(ds);
    int width  = GDALGetRasterXSize(ds);
    int height = GDALGetRasterYSize(ds);
    printf("  Bandas:%d  Tamaño:%dx%d\n", bands, width, height);

    if (bands < 4) {
        fprintf(stderr,"La imagen debe tener al menos 4 bandas (tiene %d)\n", bands);
        GDALClose(ds); exit(1);
    }

    /* Usar el primer patch de tamaño patch_size x patch_size */
    int h = (height < patch_size) ? height : patch_size;
    int w = (width  < patch_size) ? width  : patch_size;

    Tensor4D *t = tensor_create(1, 4, h, w);
    float *row  = (float*)malloc(w * sizeof(float));

    for (int b = 0; b < 4; b++) {
        GDALRasterBandH band = GDALGetRasterBand(ds, b+1);
        for (int r = 0; r < h; r++) {
            GDALRasterIO(band, GF_Read, 0, r, w, 1,
                         row, w, 1, GDT_Float32, 0, 0);
            for (int c = 0; c < w; c++) {
                float v = row[c] / 12500.0f;
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                tset(t, 0, b, r, c, v);
            }
        }
    }
    free(row);
    GDALClose(ds);
    return t;
}
#endif /* WITH_GDAL */


/* ================================================================
 * SECCIÓN 8 — GUARDAR PREDICCIÓN
 * ================================================================ */

/* Guarda el mapa de probabilidades de incendio (canal 1) como binario */
static void save_pred_bin(const Tensor4D *prob, const char *path) {
    int H = prob->height, W = prob->width;
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr,"No se pudo crear %s\n", path); return; }
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float v = tget(prob, 0, 1, h, w);
        fwrite(&v, sizeof(float), 1, f);
    }
    fclose(f);
    printf("Predicción guardada en: %s  [%dx%d floats]\n", path, H, W);
}

/* Guarda máscara binaria como archivo de texto (0/1 por píxel) */
static void save_mask_txt(const Tensor4D *prob, const char *path,
                          float threshold) {
    int H = prob->height, W = prob->width;
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# Mascara binaria incendio  threshold=%.2f\n", threshold);
    fprintf(f, "# Filas=%d Cols=%d\n", H, W);
    int fire = 0;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int v = tget(prob, 0, 1, h, w) > threshold ? 1 : 0;
            fire += v;
            fprintf(f, "%d", v);
            if (w < W-1) fputc(' ', f);
        }
        fputc('\n', f);
    }
    fclose(f);
    printf("Máscara guardada en: %s  (incendio: %d / %d píxeles)\n",
           path, fire, H*W);
}


/* ================================================================
 * SECCIÓN 9 — VALIDACIÓN NUMÉRICA
 *
 * Compara la salida C con la referencia generada por PyTorch.
 * Imprime el error absoluto medio y máximo.
 * ================================================================ */
static void validate_against_pytorch(const Tensor4D *prob,
                                     const char *ref_path) {
    int H = prob->height, W = prob->width;
    int n = 2 * H * W;   /* 2 canales */

    FILE *f = fopen(ref_path, "rb");
    if (!f) {
        printf("[Validación] No se encontró referencia: %s\n", ref_path);
        return;
    }
    float *ref = (float*)malloc(n * sizeof(float));
    size_t nr2 = fread(ref, sizeof(float), n, f); (void)nr2;
    fclose(f);

    /* La referencia de PyTorch tiene forma [1, 2, H, W] aplanada.
     * prob también es [1, 2, H, W]. Comparar elemento a elemento. */
    double mae = 0.0, max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)prob->data[i] - (double)ref[i]);
        mae += err;
        if (err > max_err) max_err = err;
    }
    mae /= n;

    printf("\n=== Validación contra PyTorch ===\n");
    printf("  Error absoluto medio (MAE): %.8f\n", mae);
    printf("  Error máximo:              %.8f\n", max_err);
    if (max_err < 1e-4)
        printf("  ✓ Coincidencia numérica correcta\n");
    else if (max_err < 1e-2)
        printf("  ~ Pequeñas diferencias (precisión float aceptable)\n");
    else
        printf("  ✗ Diferencias grandes — revisar implementación\n");

    free(ref);
}

/* Imprime estadísticas del mapa de probabilidades de incendio */
static void print_stats(const Tensor4D *prob, float threshold) {
    int H = prob->height, W = prob->width;
    float mn = 1e30f, mx = -1e30f, sum = 0.0f;
    int fire = 0;
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float v = tget(prob, 0, 1, h, w);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        if (v > threshold) fire++;
    }
    printf("\n=== Estadísticas de predicción (canal incendio) ===\n");
    printf("  Tamaño:       %d x %d\n", H, W);
    printf("  Probabilidad  min=%.6f  max=%.6f  mean=%.6f\n",
           mn, mx, sum/(H*W));
    printf("  Umbral 0.5:   %d / %d píxeles (%.2f%%)\n",
           fire, H*W, 100.0f*fire/(H*W));
}


/* ================================================================
 * SECCIÓN 10 — MAIN
 * ================================================================ */
static void print_usage(const char *prog) {
    printf("Uso:\n");
    printf("  Validación numérica:\n");
    printf("    %s --test <weights_bin_dir>\n\n", prog);
    printf("  Predicción sobre imagen binaria:\n");
    printf("    %s --weights <dir> --image-bin <input.bin> \\\n", prog);
    printf("       --H 128 --W 128 --output pred.bin\n\n");
#ifdef WITH_GDAL
    printf("  Predicción sobre imagen TIFF:\n");
    printf("    %s --weights <dir> --image <imagen.tif> \\\n", prog);
    printf("       --size 128 --output pred.bin\n\n");
#endif
}

int main(int argc, char **argv) {
    printf("=== WildfireNet (UNet2D) — Predicción en C ===\n\n");

    if (argc < 2) { print_usage(argv[0]); return 1; }

    /* ---- Parseo de argumentos ---- */
    const char *weights_dir = "weights_bin";
    const char *image_path __attribute__((unused)) = NULL;
    const char *image_bin   = NULL;
    const char *output_path = "pred.bin";
    const char *mask_path   = "pred_mask.txt";
    int do_test = 0;
    int H = 128, W = 128;
    float threshold = 0.5f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--test") && i+1<argc) {
            do_test = 1; weights_dir = argv[++i];
        } else if (!strcmp(argv[i],"--weights") && i+1<argc) {
            weights_dir = argv[++i];
        } else if (!strcmp(argv[i],"--image") && i+1<argc) {
            image_path = argv[++i];
        } else if (!strcmp(argv[i],"--image-bin") && i+1<argc) {
            image_bin = argv[++i];
        } else if (!strcmp(argv[i],"--output") && i+1<argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i],"--mask") && i+1<argc) {
            mask_path = argv[++i];
        } else if (!strcmp(argv[i],"--H") && i+1<argc) {
            H = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--W") && i+1<argc) {
            W = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--size") && i+1<argc) {
            H = W = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--threshold") && i+1<argc) {
            threshold = (float)atof(argv[++i]);
        }
    }

    /* ---- Cargar pesos ---- */
    printf("Cargando pesos desde: %s\n", weights_dir);
    UNetParams net = load_unet_params(weights_dir);
    printf("✓ Pesos cargados\n\n");

    /* ---- Cargar imagen ---- */
    Tensor4D *input = NULL;

    if (do_test) {
        /* Modo test: cargar la entrada de prueba generada por export_weights.py */
        char shape_path[512];
        snprintf(shape_path, sizeof(shape_path), "%s/test_input_shape.txt",
                 weights_dir);
        FILE *sf = fopen(shape_path, "r");
        int tb=1, tc=4, th=128, tw=128;
        if (sf) { int r = fscanf(sf,"%d %d %d %d",&tb,&tc,&th,&tw); (void)r; fclose(sf); }
        H = th; W = tw;

        char bin_path[512];
        snprintf(bin_path, sizeof(bin_path), "%s/test_input.bin", weights_dir);
        input = load_image_bin(bin_path, tb, tc, th, tw);

    } else if (image_bin) {
        input = load_image_bin(image_bin, 1, 4, H, W);

#ifdef WITH_GDAL
    } else if (image_path) {
        input = load_image_tiff(image_path, H);
        H = input->height; W = input->width;
#endif
    } else {
        fprintf(stderr,"Error: indica --test, --image-bin o --image\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("Entrada cargada: [%d, %d, %d, %d]\n\n",
           input->batch, input->channels, input->height, input->width);

    /* ---- Forward pass ---- */
    printf("Corriendo forward pass...\n");
    Tensor4D *logits = unet_forward(input, &net);
    tensor_free(input);
    printf(" Forward pass completado\n");

    /* ---- Softmax → probabilidades ---- */
    softmax_channels(logits);   /* ahora logits contiene probabilidades */
    Tensor4D *prob = logits;

    /* ---- Estadísticas ---- */
    print_stats(prob, threshold);

    /* ---- Guardar resultados ---- */
    save_pred_bin(prob, output_path);
    save_mask_txt(prob, mask_path, threshold);

    /* ---- Validación contra PyTorch (solo en modo --test) ---- */
    if (do_test) {
        char ref_path[512];
        snprintf(ref_path, sizeof(ref_path), "%s/test_output_probs.bin",
                 weights_dir);
        validate_against_pytorch(prob, ref_path);
    }

    /* ---- Liberar memoria ---- */
    tensor_free(prob);
    free_unet_params(&net);

    printf("\n=== Predicción completada ===\n");
    return 0;
}
