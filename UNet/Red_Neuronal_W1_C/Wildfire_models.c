/*
 * unet2d.c
 *
 * Implementación completa en C de UNet2D para segmentación agrícola.
 * Fusiona convolutions_unet.c y unet_ops.c con todos los bugs corregidos:
 *
 *   CORRECCIÓN 1 — fprintf inválido en tensor_alloc:
 *     fprintf("Memoria...", ...) → printf("Memoria...", ...)
 *
 *   CORRECCIÓN 2 — ReLU prematura en transposed_conv2d:
 *     ConvTranspose no lleva activación en UNet; el bloque ReLU
 *     al final fue eliminado. La activación ocurre dentro de
 *     double_conv2d (después de cada Conv3x3 + BN).
 *
 *   CORRECCIÓN 3 — orden de concatenación invertido en up_block:
 *     concat_channels(&up, skip, ...) → concat_channels(skip, &up, ...)
 *     El encoder va primero, igual que torch.cat([encoder, decoder], dim=1).
 *
 * Arquitectura UNet2D (in_channels=4, out_channels=2):
 *
 *   Encoder:
 *     enc1: DoubleConv(4   → 64)   + MaxPool
 *     enc2: DoubleConv(64  → 128)  + MaxPool
 *     enc3: DoubleConv(128 → 256)  + MaxPool
 *     enc4: DoubleConv(256 → 512)  + MaxPool
 *
 *   Bottleneck:
 *     DoubleConv(512 → 1024)
 *
 *   Decoder:
 *     upconv4: ConvTranspose(1024 → 512)  + cat(enc4) + DoubleConv(1024 → 512)
 *     upconv3: ConvTranspose(512  → 256)  + cat(enc3) + DoubleConv(512  → 256)
 *     upconv2: ConvTranspose(256  → 128)  + cat(enc2) + DoubleConv(256  → 128)
 *     upconv1: ConvTranspose(128  → 64)   + cat(enc1) + DoubleConv(128  → 64)
 *
 *   Salida:
 *     Conv1x1(64 → 2)
 *
 * Compilar: gcc -O2 -o unet2d unet2d.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* ================================================================
 * SECCIÓN 1 — TENSOR 4D
 *
 * Representa datos con dimensiones [batch, channels, height, width].
 * Todos los elementos se almacenan en un arreglo lineal contiguo
 * en orden row-major (el último índice varía más rápido).
 *
 * Índice lineal de (b, c, h, w):
 *   b*(C*H*W) + c*(H*W) + h*W + w
 * ================================================================ */
typedef struct {
    float *data;
    int batch, channels, height, width;
} Tensor4D;

/*
 * tensor_create
 * Reserva memoria e inicializa a cero con calloc.
 * La inicialización a cero es necesaria para que las sumas de
 * convolución transpuesta partan de un estado limpio.
 */
Tensor4D* tensor_create(int batch, int channels, int height, int width) {
    Tensor4D *t  = (Tensor4D*)malloc(sizeof(Tensor4D));
    t->batch     = batch;
    t->channels  = channels;
    t->height    = height;
    t->width     = width;
    t->data      = (float*)calloc((size_t)batch * channels * height * width,
                                  sizeof(float));
    if (!t->data) {
        /* CORRECCIÓN 1: usar printf en lugar de fprintf sin stream */
        printf("[Error] tensor_create: sin memoria para (%d,%d,%d,%d)\n",
               batch, channels, height, width);
        free(t);
        return NULL;
    }
    return t;
}

/* Lectura del elemento (b, c, h, w) — inline para evitar overhead en bucles */
static inline float tensor_get(const Tensor4D *t,
                                int b, int c, int h, int w) {
    return t->data[b*(t->channels*t->height*t->width)
                  + c*(t->height*t->width)
                  + h*t->width
                  + w];
}

/* Escritura del elemento (b, c, h, w) */
static inline void tensor_set(Tensor4D *t,
                               int b, int c, int h, int w,
                               float val) {
    t->data[b*(t->channels*t->height*t->width)
           + c*(t->height*t->width)
           + h*t->width
           + w] = val;
}

/* Libera la memoria del tensor */
void tensor_free(Tensor4D *t) {
    if (t) { free(t->data); free(t); }
}


/* ================================================================
 * SECCIÓN 2 — CONVOLUCIÓN 3×3 CON PADDING=1
 *
 * Parámetros fijos: kernel=3, padding=1, stride=1.
 * El padding conserva H y W en la salida (modo "same").
 *
 * Peso [C_out, C_in, 3, 3] en orden row-major.
 * Índice: co*(C_in*9) + ci*(9) + ky*3 + kx
 * ================================================================ */
void conv2d_3x3(const Tensor4D *input,
                const float    *weight,
                const float    *bias,
                Tensor4D       *output)
{
    const int B     = input->batch;
    const int C_in  = input->channels;
    const int H     = input->height;
    const int W     = input->width;
    const int C_out = output->channels;
    const int KH = 3, KW = 3, pad = 1, stride = 1;

    for (int b  = 0; b  < B;     b++)
    for (int co = 0; co < C_out; co++)
    for (int oh = 0; oh < H;     oh++)
    for (int ow = 0; ow < W;     ow++) {

        float sum = (bias != NULL) ? bias[co] : 0.0f;

        for (int ci = 0; ci < C_in; ci++)
        for (int ky = 0; ky < KH;   ky++)
        for (int kx = 0; kx < KW;   kx++) {

            int ih = oh * stride - pad + ky;
            int iw = ow * stride - pad + kx;

            /* Zero-padding implícito: solo acumula si está dentro */
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int w_idx = co*(C_in*KH*KW)
                          + ci*(KH*KW)
                          + ky*KW
                          + kx;
                sum += tensor_get(input, b, ci, ih, iw) * weight[w_idx];
            }
        }

        tensor_set(output, b, co, oh, ow, sum);
    }
}


/* ================================================================
 * SECCIÓN 3 — CONVOLUCIÓN 1×1 (capa de salida)
 *
 * Caso especial de conv2d sin padding ni bucles de kernel.
 * Mapea 64 canales → out_channels (típicamente 2) píxel a píxel.
 *
 * Peso [C_out, C_in, 1, 1] → índice: co*C_in + ci
 * ================================================================ */
void conv2d_1x1(const Tensor4D *input,
                const float    *weight,
                const float    *bias,
                Tensor4D       *output)
{
    const int B     = input->batch;
    const int C_in  = input->channels;
    const int H     = input->height;
    const int W     = input->width;
    const int C_out = output->channels;

    for (int b  = 0; b  < B;     b++)
    for (int co = 0; co < C_out; co++)
    for (int oh = 0; oh < H;     oh++)
    for (int ow = 0; ow < W;     ow++) {

        float sum = (bias != NULL) ? bias[co] : 0.0f;

        for (int ci = 0; ci < C_in; ci++)
            sum += tensor_get(input, b, ci, oh, ow) * weight[co*C_in + ci];

        tensor_set(output, b, co, oh, ow, sum);
    }
}


/* ================================================================
 * SECCIÓN 4 — RELU IN-PLACE
 *
 * Aplica max(0, x) elemento a elemento sobre todo el tensor.
 * Opera sobre el arreglo lineal directamente porque ReLU no
 * depende de la posición en ninguna dimensión.
 * ================================================================ */
void relu_inplace(Tensor4D *t) {
    int total = t->batch * t->channels * t->height * t->width;
    for (int i = 0; i < total; i++)
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
}


/* ================================================================
 * SECCIÓN 5 — BATCH NORMALIZATION 2D (modo inferencia)
 *
 * Normaliza cada canal con la media y varianza acumuladas durante
 * el entrenamiento, luego reescala con gamma y beta aprendidos.
 *
 * Fórmula: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * inv_std se calcula una vez por canal para evitar repetir la
 * raíz cuadrada en cada píxel (puede haber millones de píxeles).
 * ================================================================ */
void batch_norm2d(Tensor4D    *t,
                  const float *gamma,
                  const float *beta,
                  const float *mean,
                  const float *var,
                  float        eps)
{
    for (int b = 0; b < t->batch;    b++)
    for (int c = 0; c < t->channels; c++) {
        float inv_std = 1.0f / sqrtf(var[c] + eps);
        for (int h = 0; h < t->height; h++)
        for (int w = 0; w < t->width;  w++) {
            float x = tensor_get(t, b, c, h, w);
            tensor_set(t, b, c, h, w,
                       gamma[c] * (x - mean[c]) * inv_std + beta[c]);
        }
    }
}


/* ================================================================
 * SECCIÓN 6 — BLOQUE DOUBLE CONV
 *
 * Reproduce el bloque DoubleConv2D de PyTorch:
 *   Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
 *
 * mid_channels es el número de canales de salida de la primera
 * convolución (= out_channels en todos los bloques de UNet2D).
 *
 * Tensores intermedios se crean y liberan aquí para no exponer
 * memoria temporal al llamador.
 * ================================================================ */
void double_conv2d(const Tensor4D *input,
                   int             mid_channels,
                   int             out_channels __attribute__((unused)),
                   const float *w1,     const float *b1,
                   const float *gamma1, const float *beta1,
                   const float *mean1,  const float *var1,
                   const float *w2,     const float *b2,
                   const float *gamma2, const float *beta2,
                   const float *mean2,  const float *var2,
                   float        eps,
                   Tensor4D    *output)
{
    /* Tensor intermedio: salida de la primera Conv3x3 */
    Tensor4D *mid = tensor_create(input->batch,
                                  mid_channels,
                                  input->height,
                                  input->width);

    /* Primera pasada: Conv3x3 → BN → ReLU */
    conv2d_3x3(input, w1, b1, mid);
    batch_norm2d(mid, gamma1, beta1, mean1, var1, eps);
    relu_inplace(mid);

    /* Segunda pasada: Conv3x3 → BN → ReLU */
    conv2d_3x3(mid, w2, b2, output);
    batch_norm2d(output, gamma2, beta2, mean2, var2, eps);
    relu_inplace(output);

    tensor_free(mid);
}


/* ================================================================
 * SECCIÓN 7 — MAX POOL 2×2
 *
 * Reduce H y W a la mitad seleccionando el máximo de cada
 * ventana 2×2 sin solapamiento (stride=2).
 * Opera canal por canal: MaxPool no mezcla información entre
 * canales, solo comprime el mapa espacial de cada uno.
 * ================================================================ */
Tensor4D* maxpool2d(const Tensor4D *input) {
    const int B  = input->batch;
    const int C  = input->channels;
    const int H  = input->height;
    const int W  = input->width;

    if (H % 2 != 0 || W % 2 != 0) {
        printf("[Error] maxpool2d: H=%d o W=%d no son pares\n", H, W);
        return NULL;
    }

    Tensor4D *out = tensor_create(B, C, H/2, W/2);

    for (int b  = 0; b  < B;   b++)
    for (int c  = 0; c  < C;   c++)
    for (int oh = 0; oh < H/2; oh++)
    for (int ow = 0; ow < W/2; ow++) {

        int ih0 = oh * 2, iw0 = ow * 2;

        /* Primer elemento de la ventana 2×2 como máximo inicial */
        float mx = tensor_get(input, b, c, ih0, iw0);

        /* Recorrer los 4 elementos y quedarse con el mayor */
        for (int ki = 0; ki < 2; ki++)
        for (int kj = 0; kj < 2; kj++) {
            float v = tensor_get(input, b, c, ih0+ki, iw0+kj);
            if (v > mx) mx = v;
        }

        tensor_set(out, b, c, oh, ow, mx);
    }
    return out;
}


/* ================================================================
 * SECCIÓN 8 — CONVOLUCIÓN TRANSPUESTA 2×2 (upsampling)
 *
 * Duplica H y W: cada píxel de entrada "dispersa" su valor a un
 * bloque 2×2 en la salida, ponderado por el kernel.
 *
 * Parámetros fijos: kernel=2, stride=2, padding=0.
 * Peso [C_in, C_out, 2, 2] (ejes de canal invertidos respecto a
 * Conv2d, igual que nn.ConvTranspose2d en PyTorch).
 * Índice: ci*(C_out*4) + co*(4) + ky*2 + kx
 *
 * CORRECCIÓN 2: se eliminó el bloque ReLU que había al final.
 * nn.ConvTranspose2d no aplica activación; esta aparece dentro
 * de double_conv2d, después de cada Conv3x3 + BN.
 * ================================================================ */
Tensor4D* conv_transpose2d_2x2(const Tensor4D *input,
                                const float    *weight,
                                const float    *bias,
                                int             C_out)
{
    const int B     = input->batch;
    const int C_in  = input->channels;
    const int H_in  = input->height;
    const int W_in  = input->width;
    const int KH = 2, KW = 2, stride = 2;

    Tensor4D *output = tensor_create(B, C_out, H_in*stride, W_in*stride);

    /* Pre-cargar bias en cada posición de salida.
     * Es obligatorio antes de acumular porque múltiples píxeles de
     * entrada escriben sobre el mismo píxel de salida. */
    if (bias != NULL) {
        for (int b  = 0; b  < B;              b++)
        for (int co = 0; co < C_out;          co++)
        for (int oh = 0; oh < output->height; oh++)
        for (int ow = 0; ow < output->width;  ow++)
            tensor_set(output, b, co, oh, ow, bias[co]);
    }
    /* Si no hay bias, tensor_create ya inicializó todo a 0 con calloc */

    /* Para cada píxel de entrada: distribuir hacia la salida */
    for (int b  = 0; b  < B;    b++)
    for (int ci = 0; ci < C_in; ci++)
    for (int ih = 0; ih < H_in; ih++)
    for (int iw = 0; iw < W_in; iw++) {

        float in_val = tensor_get(input, b, ci, ih, iw);

        for (int co = 0; co < C_out; co++)
        for (int ky = 0; ky < KH;   ky++)
        for (int kx = 0; kx < KW;   kx++) {

            int oh = ih * stride + ky;
            int ow = iw * stride + kx;

            int w_idx = ci*(C_out*KH*KW)
                      + co*(KH*KW)
                      + ky*KW
                      + kx;

            float prev = tensor_get(output, b, co, oh, ow);
            tensor_set(output, b, co, oh, ow,
                       prev + in_val * weight[w_idx]);
        }
    }

    /*
     * CORRECCIÓN 2: NO se aplica ReLU aquí.
     * En el modelo PyTorch nn.ConvTranspose2d no lleva activación.
     * La activación (ReLU) ocurre solo dentro de double_conv2d.
     */
    return output;
}


/* ================================================================
 * SECCIÓN 9 — SKIP CONNECTION (concatenación por canales)
 *
 * Une dos tensores a lo largo del eje de canales.
 * Equivale a torch.cat([encoder, decoder], dim=1).
 *
 * CORRECCIÓN 3: el orden correcto es (encoder, decoder):
 *   - Los canales del encoder van en posiciones 0..enc->channels-1
 *   - Los canales del decoder van en posiciones enc->channels..total-1
 *
 * Ambos tensores deben tener el mismo B, H y W.
 * ================================================================ */
Tensor4D* concat_channels(const Tensor4D *enc, const Tensor4D *dec) {
    if (enc->batch  != dec->batch  ||
        enc->height != dec->height ||
        enc->width  != dec->width) {
        printf("[Error] concat_channels: dimensiones incompatibles "
               "enc[%d,%d,%d,%d] vs dec[%d,%d,%d,%d]\n",
               enc->batch, enc->channels, enc->height, enc->width,
               dec->batch, dec->channels, dec->height, dec->width);
        return NULL;
    }

    Tensor4D *out = tensor_create(enc->batch,
                                  enc->channels + dec->channels,
                                  enc->height,
                                  enc->width);

    for (int b = 0; b < enc->batch; b++) {

        /* Canales del encoder en posiciones 0..enc->channels-1 */
        for (int c = 0; c < enc->channels; c++)
        for (int h = 0; h < enc->height;   h++)
        for (int w = 0; w < enc->width;    w++)
            tensor_set(out, b, c, h, w,
                       tensor_get(enc, b, c, h, w));

        /* Canales del decoder desplazados a partir de enc->channels */
        for (int c = 0; c < dec->channels; c++)
        for (int h = 0; h < enc->height;   h++)
        for (int w = 0; w < enc->width;    w++)
            tensor_set(out, b, enc->channels + c, h, w,
                       tensor_get(dec, b, c, h, w));
    }

    return out;
}


/* ================================================================
 * SECCIÓN 10 — PARÁMETROS DE BLOQUE DOUBLECONV
 *
 * Agrupa todos los pesos y estadísticas de un bloque DoubleConv2D.
 * Facilita pasar los parámetros a double_conv2d sin listas largas.
 * ================================================================ */
typedef struct {
    /* Primera Conv3x3 */
    const float *w1, *b1, *gamma1, *beta1, *mean1, *var1;
    /* Segunda Conv3x3 */
    const float *w2, *b2, *gamma2, *beta2, *mean2, *var2;
    int  mid_channels;  /* canales de salida de la primera conv   */
    int  out_channels;  /* canales de salida de la segunda conv   */
    float eps;          /* epsilon para BatchNorm (típico: 1e-5)  */
} DoubleConvParams;


/* ================================================================
 * SECCIÓN 11 — PARÁMETROS DE BLOQUE DECODER (UP BLOCK)
 *
 * Cada paso del decoder realiza:
 *   1. ConvTranspose2×2 (upsampling)
 *   2. Concatenación con skip del encoder (CORRECCIÓN 3: enc primero)
 *   3. DoubleConv sobre la concatenación
 * ================================================================ */
typedef struct {
    /* Pesos de la convolución transpuesta */
    const float *up_weight, *up_bias;
    int          C_up;       /* canales de salida del upsampling */

    /* Parámetros del DoubleConv posterior */
    DoubleConvParams dconv;
} UpBlockParams;


/* ================================================================
 * SECCIÓN 12 — BLOQUE DECODER COMPLETO
 *
 * Ejecuta un paso completo del decoder:
 *   x (decoder anterior) → ConvTranspose → cat con enc_skip → DoubleConv → salida
 *
 * CORRECCIÓN 3 aplicada aquí: concat_channels(enc_skip, up, ...)
 * pone el tensor del encoder primero, igual que PyTorch.
 * ================================================================ */
Tensor4D* up_block(const Tensor4D     *x,
                   const Tensor4D     *enc_skip,
                   const UpBlockParams *p)
{
    /* Paso 1: duplicar resolución espacial */
    Tensor4D *up = conv_transpose2d_2x2(x, p->up_weight, p->up_bias, p->C_up);
    if (!up) return NULL;

    /* Paso 2: concatenar — CORRECCIÓN 3: encoder primero
     * Equivale a torch.cat([enc_skip, up], dim=1) */
    Tensor4D *cat = concat_channels(enc_skip, up);
    tensor_free(up);
    if (!cat) return NULL;

    /* Paso 3: DoubleConv sobre la concatenación */
    Tensor4D *out = tensor_create(cat->batch,
                                  p->dconv.out_channels,
                                  cat->height,
                                  cat->width);
    if (!out) { tensor_free(cat); return NULL; }

    double_conv2d(cat,
                  p->dconv.mid_channels,
                  p->dconv.out_channels,
                  p->dconv.w1, p->dconv.b1,
                  p->dconv.gamma1, p->dconv.beta1,
                  p->dconv.mean1,  p->dconv.var1,
                  p->dconv.w2, p->dconv.b2,
                  p->dconv.gamma2, p->dconv.beta2,
                  p->dconv.mean2,  p->dconv.var2,
                  p->dconv.eps,
                  out);

    tensor_free(cat);
    return out;
}


/* ================================================================
 * SECCIÓN 13 — FORWARD PASS COMPLETO DE UNET2D
 *
 * Recibe la imagen de entrada y todos los pesos del modelo.
 * Devuelve el mapa de segmentación de salida.
 *
 * Todos los tensores intermedios se crean y liberan aquí;
 * el llamador solo recibe el tensor de salida y es responsable
 * de liberarlo con tensor_free cuando ya no lo necesite.
 * ================================================================ */

/* Parámetros completos del modelo */
typedef struct {
    /* Encoder */
    DoubleConvParams enc1, enc2, enc3, enc4;
    /* Bottleneck */
    DoubleConvParams bottleneck;
    /* Decoder */
    UpBlockParams    up4, up3, up2, up1;
    /* Capa de salida Conv1×1 */
    const float *out_weight, *out_bias;
    int          out_channels;  /* típicamente 2 */
} UNet2DParams;

Tensor4D* unet2d_forward(const Tensor4D *input, const UNet2DParams *net) {
    /* ---- ENCODER ---- */

    /* enc1: [B, 4, H, W] → [B, 64, H, W] */
    Tensor4D *e1 = tensor_create(input->batch, net->enc1.out_channels,
                                 input->height, input->width);
    double_conv2d(input,
                  net->enc1.mid_channels, net->enc1.out_channels,
                  net->enc1.w1, net->enc1.b1,
                  net->enc1.gamma1, net->enc1.beta1,
                  net->enc1.mean1,  net->enc1.var1,
                  net->enc1.w2, net->enc1.b2,
                  net->enc1.gamma2, net->enc1.beta2,
                  net->enc1.mean2,  net->enc1.var2,
                  net->enc1.eps, e1);

    /* pool1: [B, 64, H, W] → [B, 64, H/2, W/2] */
    Tensor4D *p1 = maxpool2d(e1);

    /* enc2: [B, 64, H/2, W/2] → [B, 128, H/2, W/2] */
    Tensor4D *e2 = tensor_create(p1->batch, net->enc2.out_channels,
                                 p1->height, p1->width);
    double_conv2d(p1,
                  net->enc2.mid_channels, net->enc2.out_channels,
                  net->enc2.w1, net->enc2.b1,
                  net->enc2.gamma1, net->enc2.beta1,
                  net->enc2.mean1,  net->enc2.var1,
                  net->enc2.w2, net->enc2.b2,
                  net->enc2.gamma2, net->enc2.beta2,
                  net->enc2.mean2,  net->enc2.var2,
                  net->enc2.eps, e2);
    tensor_free(p1);

    /* pool2: [B, 128, H/2, W/2] → [B, 128, H/4, W/4] */
    Tensor4D *p2 = maxpool2d(e2);

    /* enc3: [B, 128, H/4, W/4] → [B, 256, H/4, W/4] */
    Tensor4D *e3 = tensor_create(p2->batch, net->enc3.out_channels,
                                 p2->height, p2->width);
    double_conv2d(p2,
                  net->enc3.mid_channels, net->enc3.out_channels,
                  net->enc3.w1, net->enc3.b1,
                  net->enc3.gamma1, net->enc3.beta1,
                  net->enc3.mean1,  net->enc3.var1,
                  net->enc3.w2, net->enc3.b2,
                  net->enc3.gamma2, net->enc3.beta2,
                  net->enc3.mean2,  net->enc3.var2,
                  net->enc3.eps, e3);
    tensor_free(p2);

    /* pool3: [B, 256, H/4, W/4] → [B, 256, H/8, W/8] */
    Tensor4D *p3 = maxpool2d(e3);

    /* enc4: [B, 256, H/8, W/8] → [B, 512, H/8, W/8] */
    Tensor4D *e4 = tensor_create(p3->batch, net->enc4.out_channels,
                                 p3->height, p3->width);
    double_conv2d(p3,
                  net->enc4.mid_channels, net->enc4.out_channels,
                  net->enc4.w1, net->enc4.b1,
                  net->enc4.gamma1, net->enc4.beta1,
                  net->enc4.mean1,  net->enc4.var1,
                  net->enc4.w2, net->enc4.b2,
                  net->enc4.gamma2, net->enc4.beta2,
                  net->enc4.mean2,  net->enc4.var2,
                  net->enc4.eps, e4);
    tensor_free(p3);

    /* pool4: [B, 512, H/8, W/8] → [B, 512, H/16, W/16] */
    Tensor4D *p4 = maxpool2d(e4);

    /* ---- BOTTLENECK ---- */
    /* [B, 512, H/16, W/16] → [B, 1024, H/16, W/16] */
    Tensor4D *bn = tensor_create(p4->batch, net->bottleneck.out_channels,
                                 p4->height, p4->width);
    double_conv2d(p4,
                  net->bottleneck.mid_channels, net->bottleneck.out_channels,
                  net->bottleneck.w1, net->bottleneck.b1,
                  net->bottleneck.gamma1, net->bottleneck.beta1,
                  net->bottleneck.mean1,  net->bottleneck.var1,
                  net->bottleneck.w2, net->bottleneck.b2,
                  net->bottleneck.gamma2, net->bottleneck.beta2,
                  net->bottleneck.mean2,  net->bottleneck.var2,
                  net->bottleneck.eps, bn);
    tensor_free(p4);

    /* ---- DECODER ---- */

    /* d4: [B, 1024, H/16, W/16] → ConvT → cat(e4) → DoubleConv → [B, 512, H/8, W/8]
     * CORRECCIÓN 3: concat_channels pone e4 primero dentro de up_block */
    Tensor4D *d4 = up_block(bn, e4, &net->up4);
    tensor_free(bn);
    tensor_free(e4);

    /* d3: → [B, 256, H/4, W/4] */
    Tensor4D *d3 = up_block(d4, e3, &net->up3);
    tensor_free(d4);
    tensor_free(e3);

    /* d2: → [B, 128, H/2, W/2] */
    Tensor4D *d2 = up_block(d3, e2, &net->up2);
    tensor_free(d3);
    tensor_free(e2);

    /* d1: → [B, 64, H, W] */
    Tensor4D *d1 = up_block(d2, e1, &net->up1);
    tensor_free(d2);
    tensor_free(e1);

    /* ---- SALIDA Conv1×1 ---- */
    /* [B, 64, H, W] → [B, out_channels, H, W] */
    Tensor4D *out = tensor_create(d1->batch, net->out_channels,
                                  d1->height, d1->width);
    conv2d_1x1(d1, net->out_weight, net->out_bias, out);
    tensor_free(d1);

    return out;
}


/* ================================================================
 * SECCIÓN 14 — UTILIDADES PARA EL DEMO
 * ================================================================ */

/* Rellena un arreglo con un valor constante */
static void fill_const(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) arr[i] = val;
}

/* Reserva y rellena un arreglo en un solo paso */
static float* make_weights(int n, float val) {
    float *w = (float*)malloc((size_t)n * sizeof(float));
    fill_const(w, n, val);
    return w;
}

/* Reserva y rellena arreglos de BN (gamma=1, beta=0, mean=0, var=1) */
static float* make_ones (int n) { return make_weights(n, 1.0f); }
static float* make_zeros(int n) { return make_weights(n, 0.0f); }

/*
 * make_double_conv_params
 * Crea parámetros de juguete para un bloque DoubleConv.
 * En un modelo real estos vendrían de un archivo de pesos entrenado.
 *
 *   c_in  → número de canales de entrada
 *   c_out → número de canales de salida (mid_channels = out_channels aquí)
 */
static DoubleConvParams make_double_conv_params(int c_in, int c_out) {
    DoubleConvParams p;
    p.mid_channels = c_out;
    p.out_channels = c_out;
    p.eps          = 1e-5f;

    /* Pesos Conv3x3 escalados pequeños para evitar explosión numérica */
    p.w1 = make_weights(c_out * c_in  * 9, 0.001f);
    p.b1 = make_zeros(c_out);
    p.w2 = make_weights(c_out * c_out * 9, 0.001f);
    p.b2 = make_zeros(c_out);

    /* BatchNorm en identidad: gamma=1, beta=0, mean=0, var=1 */
    p.gamma1 = make_ones (c_out);
    p.beta1  = make_zeros(c_out);
    p.mean1  = make_zeros(c_out);
    p.var1   = make_ones (c_out);
    p.gamma2 = make_ones (c_out);
    p.beta2  = make_zeros(c_out);
    p.mean2  = make_zeros(c_out);
    p.var2   = make_ones (c_out);

    return p;
}

/* Libera todos los arreglos dentro de un DoubleConvParams */
static void free_double_conv_params(DoubleConvParams *p) {
    free((void*)p->w1);     free((void*)p->b1);
    free((void*)p->gamma1); free((void*)p->beta1);
    free((void*)p->mean1);  free((void*)p->var1);
    free((void*)p->w2);     free((void*)p->b2);
    free((void*)p->gamma2); free((void*)p->beta2);
    free((void*)p->mean2);  free((void*)p->var2);
}

/*
 * make_up_block_params
 * Crea parámetros de juguete para un bloque de decoder.
 *
 *   c_in    → canales de entrada (vienen del nivel anterior del decoder)
 *   c_up    → canales de salida del ConvTranspose (típicamente c_in/2)
 *   c_skip  → canales de la skip connection del encoder
 *   c_out   → canales de salida del DoubleConv final
 */
static UpBlockParams make_up_block_params(int c_in, int c_up,
                                          int c_skip, int c_out) {
    UpBlockParams p;
    p.C_up      = c_up;
    p.up_weight = make_weights(c_in * c_up * 4, 0.001f);  /* [c_in, c_up, 2, 2] */
    p.up_bias   = make_zeros(c_up);

    /* DoubleConv sobre la concatenación: c_up + c_skip canales de entrada */
    p.dconv = make_double_conv_params(c_up + c_skip, c_out);

    return p;
}

/* Libera todos los arreglos dentro de un UpBlockParams */
static void free_up_block_params(UpBlockParams *p) {
    free((void*)p->up_weight);
    free((void*)p->up_bias);
    free_double_conv_params(&p->dconv);
}


/* ================================================================
 * SECCIÓN 15 — DEMO PRINCIPAL
 *
 * Construye una UNet2D con pesos de juguete y ejecuta un forward
 * pass para verificar que las dimensiones son correctas y que no
 * hay errores de memoria. Los valores numéricos no son significativos
 * (pesos aleatorios pequeños), pero la arquitectura es la correcta.
 *
 * Con una entrada [1, 4, 64, 64] se esperan las dimensiones:
 *   e1:  [1,  64, 64, 64]
 *   e2:  [1, 128, 32, 32]
 *   e3:  [1, 256, 16, 16]
 *   e4:  [1, 512,  8,  8]
 *   bn:  [1,1024,  4,  4]
 *   d4:  [1, 512,  8,  8]
 *   d3:  [1, 256, 16, 16]
 *   d2:  [1, 128, 32, 32]
 *   d1:  [1,  64, 64, 64]
 *   out: [1,   2, 64, 64]
 * ================================================================ */
int main(void) {
    printf("=== UNet2D en C — Demo completo ===\n\n");

    /* Dimensiones de entrada */
    const int B = 1, C_in = 4, H = 64, W = 64;

    /* Tensor de entrada con valores de prueba */
    Tensor4D *input = tensor_create(B, C_in, H, W);
    for (int i = 0; i < B*C_in*H*W; i++)
        input->data[i] = (float)(i % 10) * 0.1f;

    printf("Entrada: [%d, %d, %d, %d]\n\n", B, C_in, H, W);

    /* ---- Construir parámetros del modelo ---- */
    UNet2DParams net;

    /* Encoder */
    net.enc1       = make_double_conv_params(4,    64);
    net.enc2       = make_double_conv_params(64,   128);
    net.enc3       = make_double_conv_params(128,  256);
    net.enc4       = make_double_conv_params(256,  512);

    /* Bottleneck */
    net.bottleneck = make_double_conv_params(512, 1024);

    /* Decoder
     * up4: entrada 1024, upconv→512, skip enc4=512, salida 512
     * up3: entrada 512,  upconv→256, skip enc3=256, salida 256
     * up2: entrada 256,  upconv→128, skip enc2=128, salida 128
     * up1: entrada 128,  upconv→64,  skip enc1=64,  salida 64  */
    net.up4 = make_up_block_params(1024, 512, 512, 512);
    net.up3 = make_up_block_params(512,  256, 256, 256);
    net.up2 = make_up_block_params(256,  128, 128, 128);
    net.up1 = make_up_block_params(128,   64,  64,  64);

    /* Capa de salida Conv1×1: 64 → 2 */
    net.out_channels = 2;
    net.out_weight   = make_weights(64 * 2, 0.001f);
    net.out_bias     = make_zeros(2);

    /* ---- Forward pass ---- */
    printf("Ejecutando forward pass...\n");
    Tensor4D *output = unet2d_forward(input, &net);

    if (output) {
        printf("\nDimensiones de salida: [%d, %d, %d, %d]\n",
               output->batch, output->channels,
               output->height, output->width);
        printf("Valor de salida [0,0,0,0] = %.6f\n", output->data[0]);
        printf("Valor de salida [0,1,0,0] = %.6f\n",
               output->data[output->height * output->width]);
        printf("\nVerificacion de dimensiones:\n");
        printf("  Esperado: [%d, %d, %d, %d]\n", B, 2, H, W);
        printf("  Obtenido: [%d, %d, %d, %d] %s\n",
               output->batch, output->channels,
               output->height, output->width,
               (output->batch == B && output->channels == 2 &&
                output->height == H && output->width == W) ? "[OK]" : "[ERROR]");
        tensor_free(output);
    } else {
        printf("[ERROR] El forward pass fallo.\n");
    }

    /* ---- Liberar parámetros ---- */
    free_double_conv_params(&net.enc1);
    free_double_conv_params(&net.enc2);
    free_double_conv_params(&net.enc3);
    free_double_conv_params(&net.enc4);
    free_double_conv_params(&net.bottleneck);
    free_up_block_params(&net.up4);
    free_up_block_params(&net.up3);
    free_up_block_params(&net.up2);
    free_up_block_params(&net.up1);
    free((void*)net.out_weight);
    free((void*)net.out_bias);

    tensor_free(input);

    printf("\n=== Demo completado ===\n");
    return 0;
}