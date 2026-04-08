/*
 * convolutions_unet.c
 *
 * Implementación en C de las convoluciones usadas en UNet2D:
 *   - Convolución 3x3     (bloque DoubleConv2D del encoder y decoder)
 *   - ConvTranspose 2x2   (upsampling del decoder)
 *   - Skip connection     (concatenación por canales entre encoder y decoder)
 *
 * Compilar: gcc -O2 -o conv_demo convolutions_unet.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* =============================================================
 * TENSOR 4D
 *
 * Representa un bloque de datos con 4 dimensiones:
 *   batch    — cuántas imágenes se procesan al mismo tiempo
 *   channels — cuántos canales tiene cada imagen
 *   height   — alto de cada imagen en píxeles
 *   width    — ancho de cada imagen en píxeles
 *
 * Todos los datos se guardan en un solo arreglo lineal (data).
 * Para acceder a un elemento se convierte (b, c, h, w) a un
 * índice lineal con tensor_get y tensor_set.
 * ============================================================= */
typedef struct {
    float *data;
    int batch, channels, height, width;
} Tensor4D;

/*
 * tensor_create
 * Reserva memoria para el tensor e inicializa todos los valores a 0.
 * Se usa calloc en lugar de malloc para garantizar la inicialización
 * a cero, necesaria para que las sumas de convolución arranquen limpias.
 */
Tensor4D* tensor_create(int batch, int channels, int height, int width) {
    Tensor4D *t  = (Tensor4D*)malloc(sizeof(Tensor4D));
    t->batch     = batch;
    t->channels  = channels;
    t->height    = height;
    t->width     = width;
    t->data      = (float*)calloc(batch * channels * height * width,
                                  sizeof(float));
    return t;
}

/*
 * tensor_get
 * Devuelve el valor en la posición (b, c, h, w) del tensor.
 *
 * Fórmula del índice lineal (row-major):
 *   idx = b*(C*H*W) + c*(H*W) + h*W + w
 *
 * Cada coordenada se multiplica por cuántos elementos hay
 * después de ella, igual que convertir horas a segundos:
 *   horas*3600 + minutos*60 + segundos
 *
 * static inline le dice al compilador que inserte este código
 * directamente donde se llama, evitando el costo de una llamada
 * a función dentro de los bucles internos.
 */
static inline float tensor_get(const Tensor4D *t,
                                int b, int c, int h, int w) {
    return t->data[b*(t->channels*t->height*t->width)
                  + c*(t->height*t->width)
                  + h*t->width
                  + w];
}

/*
 * tensor_set
 * Escribe val en la posición (b, c, h, w) del tensor.
 * Usa la misma fórmula de índice lineal que tensor_get.
 */
static inline void tensor_set(Tensor4D *t,
                               int b, int c, int h, int w,
                               float val) {
    t->data[b*(t->channels*t->height*t->width)
           + c*(t->height*t->width)
           + h*t->width
           + w] = val;
}

/*
 * tensor_free
 * Libera la memoria reservada por tensor_create.
 * Primero libera el arreglo de datos y luego la estructura.
 */
void tensor_free(Tensor4D *t) {
    if (t) { free(t->data); free(t); }
}


/* =============================================================
 * CONVOLUCIÓN 3x3
 *
 * Recorre cada píxel de salida (b, co, oh, ow) y calcula su valor
 * acumulando las multiplicaciones del kernel 3x3 sobre todos los
 * canales de entrada.
 *
 * Parámetros fijos:
 *   kernel_size = 3   ventana de 3x3 píxeles
 *   padding     = 1   borde de ceros que conserva H y W en la salida
 *   stride      = 1   el kernel se desplaza de uno en uno
 *
 * Parámetros:
 *   input   [B, C_in,  H, W]        tensor de entrada
 *   weight  [C_out, C_in, 3, 3]     pesos del kernel
 *   bias    [C_out]                  sesgo por canal (puede ser NULL)
 *   output  [B, C_out, H, W]        tensor de salida (mismo H y W)
 * ============================================================= */
void conv2d_3x3(const Tensor4D *input,
                const float    *weight,
                const float    *bias,
                Tensor4D       *output)
{
    /* Variables locales para no releer la estructura en cada iteración */
    const int B     = input->batch;
    const int C_in  = input->channels;
    const int H     = input->height;
    const int W     = input->width;
    const int C_out = output->channels;
    const int KH    = 3, KW = 3;   /* tamaño del kernel          */
    const int pad   = 1;            /* padding que conserva H y W */
    const int stride = 1;

    /*
     * Bucles externos: seleccionan a qué píxel de salida le toca.
     * Cada combinación (b, co, oh, ow) produce un único valor de salida.
     *
     *   b  — imagen dentro del batch
     *   co — canal de salida
     *   oh — fila del píxel de salida
     *   ow — columna del píxel de salida
     */
    for (int b  = 0; b  < B;     b++)
    for (int co = 0; co < C_out; co++)
    for (int oh = 0; oh < H;     oh++)
    for (int ow = 0; ow < W;     ow++) {

        /* La suma empieza en el bias del canal si existe, o en 0 */
        float sum = (bias != NULL) ? bias[co] : 0.0f;

        /*
         * Bucles internos: calculan el valor de ese píxel de salida.
         * Recorren todos los canales de entrada y la ventana 3x3
         * del kernel centrada en (oh, ow).
         *
         *   ci — canal de entrada
         *   ky — fila dentro del kernel
         *   kx — columna dentro del kernel
         */
        for (int ci = 0; ci < C_in; ci++)
        for (int ky = 0; ky < KH;   ky++)
        for (int kx = 0; kx < KW;   kx++) {

            /*
             * Calcula qué píxel de la entrada corresponde
             * a esta posición del kernel.
             * Cuando ih o iw son negativos o mayores que H/W,
             * el píxel está en la zona de padding (fuera de la imagen).
             */
            int ih = oh * stride - pad + ky;
            int iw = ow * stride - pad + kx;

            /*
             * Solo acumula si el píxel está dentro de la imagen real.
             * Si está fuera no se hace nada, equivale a multiplicar
             * por cero (zero-padding implícito).
             */
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {

                /*
                 * Índice lineal del peso para (co, ci, ky, kx).
                 * Fórmula row-major con C_out como dimensión más externa:
                 *   co*(C_in*KH*KW) — salta bloques completos de canal de salida
                 *   ci*(KH*KW)      — salta kernels completos de canal de entrada
                 *   ky*KW           — salta filas dentro del kernel
                 *   kx              — posición dentro de la fila
                 */
                int w_idx = co*(C_in*KH*KW)
                          + ci*(KH*KW)
                          + ky*KW
                          + kx;

                /* Multiplica el píxel de entrada por su peso y acumula */
                sum += tensor_get(input, b, ci, ih, iw) * weight[w_idx];
            }
        }

        /* Escribe el valor calculado en el píxel de salida correspondiente */
        tensor_set(output, b, co, oh, ow, sum);
    }
}


/* =============================================================
 * RELU
 *
 * Función de activación: si el valor es negativo lo pone a cero,
 * si es positivo lo deja igual. Se aplica elemento a elemento
 * sobre todo el tensor después de cada convolución.
 *
 * Se recorre el arreglo lineal directamente porque ReLU no
 * necesita saber en qué dimensión está cada elemento.
 * ============================================================= */
void relu_inplace(Tensor4D *t) {
    int total = t->batch * t->channels * t->height * t->width;
    for (int i = 0; i < total; i++)
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
}


/* =============================================================
 * BATCH NORMALIZATION 2D  (modo inferencia)
 *
 * Normaliza cada canal para que tenga media 0 y varianza 1,
 * luego reescala con parámetros aprendidos gamma y beta.
 *
 * Fórmula: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * inv_std se calcula una vez por canal y no dentro del bucle
 * de píxeles para evitar recalcularlo millones de veces.
 *
 * eps (típicamente 1e-5) evita división por cero si var = 0.
 *
 * Parámetros por canal (arreglos de tamaño C):
 *   gamma — escala aprendida
 *   beta  — desplazamiento aprendido
 *   mean  — media acumulada durante entrenamiento
 *   var   — varianza acumulada durante entrenamiento
 * ============================================================= */
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


/* =============================================================
 * BLOQUE DOUBLE CONV
 *
 * Reproduce el bloque DoubleConv2D de PyTorch:
 *   Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU
 *
 * Se crea un tensor intermedio (mid) para guardar el resultado
 * de la primera convolución, ya que la segunda lo necesita como
 * entrada. Al terminar se libera porque ya no se necesita.
 *
 * w1, b1, gamma1, beta1, mean1, var1 — parámetros del primer Conv
 * w2, b2, gamma2, beta2, mean2, var2 — parámetros del segundo Conv
 * ============================================================= */
void double_conv2d(const Tensor4D *input,
                   const float *w1,     const float *b1,
                   const float *gamma1, const float *beta1,
                   const float *mean1,  const float *var1,
                   const float *w2,     const float *b2,
                   const float *gamma2, const float *beta2,
                   const float *mean2,  const float *var2,
                   float        eps,
                   Tensor4D    *output)
{
    /* Tensor intermedio con la misma forma que output */
    Tensor4D *mid = tensor_create(input->batch,
                                  output->channels,
                                  input->height,
                                  input->width);

    /* Primera pasada: Conv -> BN -> ReLU */
    conv2d_3x3(input, w1, b1, mid);
    batch_norm2d(mid, gamma1, beta1, mean1, var1, eps);
    relu_inplace(mid);

    /* Segunda pasada: Conv -> BN -> ReLU */
    conv2d_3x3(mid, w2, b2, output);
    batch_norm2d(output, gamma2, beta2, mean2, var2, eps);
    relu_inplace(output);

    /* Libera el tensor intermedio */
    tensor_free(mid);
}


/* =============================================================
 * CONVOLUCIÓN TRANSPUESTA 2x2
 *
 * Operación inversa al MaxPool2d(2): duplica H y W.
 * Cada píxel de entrada distribuye su valor a un bloque 2x2
 * en la salida, ponderado por el kernel. Con stride=2 los
 * bloques de salida quedan intercalados y el mapa se duplica.
 *
 * Parámetros fijos:
 *   kernel_size = 2
 *   stride      = 2   duplica H y W
 *   padding     = 0
 *
 * Parámetros:
 *   input   [B, C_in,  H,   W  ]     tensor de entrada
 *   weight  [C_in, C_out, 2, 2]      pesos (ejes 0 y 1 invertidos
 *                                    respecto a Conv2d normal,
 *                                    así los guarda PyTorch)
 *   bias    [C_out]                   sesgo por canal (puede ser NULL)
 *   output  [B, C_out, H*2, W*2]     tensor de salida duplicado
 * ============================================================= */
void conv_transpose2d_2x2(const Tensor4D *input,
                           const float    *weight,
                           const float    *bias,
                           Tensor4D       *output)
{
    const int B     = input->batch;
    const int C_in  = input->channels;
    const int H_in  = input->height;
    const int W_in  = input->width;
    const int C_out = output->channels;
    const int stride = 2, KH = 2, KW = 2;

    /*
     * Inicialización del tensor de salida antes de los cálculos.
     * Es obligatorio hacerlo aquí porque múltiples píxeles de entrada
     * escriben sobre el mismo píxel de salida acumulando valores.
     * Sin esta inicialización se acumularía sobre basura en memoria.
     *
     * Si hay bias se precarga ese valor en cada posición.
     * Si no hay bias se pone todo a cero con memset, que es más
     * rápido que un bucle porque opera directamente sobre la memoria.
     */
    if (bias != NULL) {
        for (int b  = 0; b  < B;              b++)
        for (int co = 0; co < C_out;          co++)
        for (int oh = 0; oh < output->height; oh++)
        for (int ow = 0; ow < output->width;  ow++)
            tensor_set(output, b, co, oh, ow, bias[co]);
    } else {
        memset(output->data, 0,
               B * C_out * output->height * output->width * sizeof(float));
    }

    /*
     * Bucles externos: recorren cada píxel de ENTRADA.
     * A diferencia de Conv3x3 donde los externos recorrían la salida,
     * aquí se parte de la entrada y se distribuye hacia la salida.
     *
     * Bucles internos: para cada píxel de entrada recorren el kernel
     * 2x2 y escriben el valor ponderado en el bloque 2x2 correspondiente
     * de la salida. oh = ih*stride+ky coloca los valores intercalados
     * produciendo el efecto de duplicar la resolución.
     */
    for (int b  = 0; b  < B;    b++)
    for (int ci = 0; ci < C_in; ci++)
    for (int ih = 0; ih < H_in; ih++)
    for (int iw = 0; iw < W_in; iw++) {

        float in_val = tensor_get(input, b, ci, ih, iw);

        for (int co = 0; co < C_out; co++)
        for (int ky = 0; ky < KH;   ky++)
        for (int kx = 0; kx < KW;   kx++) {

            /* Posición en la salida donde se escribe este valor */
            int oh = ih * stride + ky;
            int ow = iw * stride + kx;

            /*
             * Índice lineal del peso para (ci, co, ky, kx).
             * ci va primero porque en PyTorch ConvTranspose2d
             * guarda los pesos con los ejes de canal invertidos
             * respecto a Conv2d: aquí el canal de entrada es la
             * dimensión más externa, no el canal de salida.
             *
             *   ci*(C_out*KH*KW) — salta bloques de canal de entrada
             *   co*(KH*KW)       — salta kernels de canal de salida
             *   ky*KW            — salta filas dentro del kernel
             *   kx               — posición dentro de la fila
             */
            int w_idx = ci*(C_out*KH*KW)
                      + co*(KH*KW)
                      + ky*KW
                      + kx;

            /* Acumula sobre el valor que ya había en esa posición */
            float prev = tensor_get(output, b, co, oh, ow);
            tensor_set(output, b, co, oh, ow,
                       prev + in_val * weight[w_idx]);
        }
    }
}


/* =============================================================
 * SKIP CONNECTION — concatenación por canales
 *
 * Une el tensor del encoder (e) con el del decoder (d) a lo largo
 * del eje de canales. Equivale a torch.cat([e, d], dim=1).
 *
 * El primer bloque de bucles copia los canales de e al tensor de
 * salida en las posiciones 0 hasta e->channels-1.
 *
 * El segundo bloque copia los canales de d a continuación, usando
 * e->channels + c como índice de destino para que los canales de d
 * empiecen justo donde terminaron los de e y no los sobreescriban.
 *
 * Requiere que e y d tengan el mismo B, H y W.
 * La salida tiene C = e->channels + d->channels.
 * ============================================================= */
Tensor4D* concat_channels(const Tensor4D *e, const Tensor4D *d) {
    Tensor4D *out = tensor_create(e->batch,
                                  e->channels + d->channels,
                                  e->height,
                                  e->width);

    for (int b = 0; b < e->batch; b++) {

        /* Primer bloque: canales de e en posiciones 0..e->channels-1 */
        for (int c = 0; c < e->channels; c++)
        for (int h = 0; h < e->height;   h++)
        for (int w = 0; w < e->width;    w++)
            tensor_set(out, b, c, h, w,
                       tensor_get(e, b, c, h, w));

        /* Segundo bloque: canales de d en posiciones e->channels..total-1
         * e->channels + c desplaza el índice para no sobreescribir lo
         * que ya copió el primer bloque                                  */
        for (int c = 0; c < d->channels; c++)
        for (int h = 0; h < e->height;   h++)
        for (int w = 0; w < e->width;    w++)
            tensor_set(out, b, e->channels + c, h, w,
                       tensor_get(d, b, c, h, w));
    }

    return out;
}


/* =============================================================
 * DEMO
 *
 * Muestra que las funciones producen las dimensiones correctas.
 * Los pesos se inicializan a 0.01 solo para tener valores con
 * qué probar. En una red real vendrían de un archivo entrenado.
 * ============================================================= */
int main(void) {
    printf("=== Demo Conv3x3 + ConvTranspose2x2 ===\n\n");

    int B = 1, C_in = 4, H = 8, W = 8, C_mid = 64, C_up = 32;

    /* Tensor de entrada con valores de prueba simples */
    Tensor4D *input = tensor_create(B, C_in, H, W);
    for (int i = 0; i < B*C_in*H*W; i++)
        input->data[i] = (float)(i % 10) * 0.1f;

    /* Conv3x3 */
    Tensor4D *after_conv = tensor_create(B, C_mid, H, W);
    int w1_size = C_mid * C_in * 3 * 3;
    float *w1 = (float*)malloc(w1_size * sizeof(float));
    float *b1 = (float*)calloc(C_mid, sizeof(float));
    for (int i = 0; i < w1_size; i++) w1[i] = 0.01f;

    conv2d_3x3(input, w1, b1, after_conv);
    relu_inplace(after_conv);

    printf("Conv3x3:\n");
    printf("  Entrada: [%d, %d, %d, %d]\n", B, C_in, H, W);
    printf("  Salida:  [%d, %d, %d, %d]  (H y W conservados)\n\n",
           B, C_mid, after_conv->height, after_conv->width);

    /* ConvTranspose2x2 */
    Tensor4D *after_up = tensor_create(B, C_up, H*2, W*2);
    int w2_size = C_mid * C_up * 2 * 2;
    float *w2 = (float*)malloc(w2_size * sizeof(float));
    float *b2 = (float*)calloc(C_up, sizeof(float));
    for (int i = 0; i < w2_size; i++) w2[i] = 0.01f;

    conv_transpose2d_2x2(after_conv, w2, b2, after_up);

    printf("ConvTranspose2x2:\n");
    printf("  Entrada: [%d, %d, %d, %d]\n", B, C_mid, H, W);
    printf("  Salida:  [%d, %d, %d, %d]  (H y W duplicados)\n\n",
           B, C_up, after_up->height, after_up->width);

    tensor_free(input);
    tensor_free(after_conv);
    tensor_free(after_up);
    free(w1); free(b1);
    free(w2); free(b2);

    printf("=== Fin ===\n");
    return 0;
}
