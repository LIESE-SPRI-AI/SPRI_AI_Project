/*
 * fully_connected.c
 *
 * Implementación en C de una capa Fully Connected (densa).
 * Es independiente del código de convoluciones.
 *
 * Incluye:
 *   - Capa Fully Connected   y = x * W^T + b
 *   - ReLU para vectores planos
 *   - Softmax para convertir logits a probabilidades
 *
 * Compilar: gcc -O2 -o fc_demo fully_connected.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* =============================================================
 * FULLY CONNECTED
 *
 * Cada neurona de salida toma TODOS los valores de entrada y los
 * combina con sus propios pesos. Es una multiplicación de matrices:
 *
 *   y = x * W^T + b
 *
 * A diferencia de la convolución que opera sobre una ventana local
 * de píxeles, aquí cada neurona de salida está conectada a cada
 * elemento de la entrada, por eso se llama "fully connected".
 *
 * La entrada debe ser un vector plano, no un tensor con dimensiones
 * espaciales H y W. Si tienes un tensor 4D debes aplanarlo primero
 * (operación flatten) antes de pasarlo a esta función.
 *
 * Parámetros:
 *   x        [B x in_feat]           vector de entrada aplanado
 *   weight   [out_feat x in_feat]    matriz de pesos
 *   bias     [out_feat]              sesgo por neurona (puede ser NULL)
 *   output   [B x out_feat]          vector de salida
 *   B        tamaño del batch
 *   in_feat  número de entradas  (neuronas de la capa anterior)
 *   out_feat número de salidas   (neuronas de esta capa)
 * ============================================================= */
void fully_connected(const float *x,
                     const float *weight,
                     const float *bias,
                     float       *output,
                     int B, int in_feat, int out_feat)
{
    /*
     * Bucle externo: recorre cada imagen del batch.
     * Cada imagen tiene su propio vector de entrada independiente.
     */
    for (int b = 0; b < B; b++)

    /*
     * Bucle medio: recorre cada neurona de salida.
     * Cada neurona produce un único valor calculando la suma
     * ponderada de todos los valores de entrada.
     */
    for (int o = 0; o < out_feat; o++) {

        /* La suma empieza en el bias de esa neurona si existe */
        float sum = (bias != NULL) ? bias[o] : 0.0f;

        /*
         * Bucle interno: recorre todos los valores de entrada.
         * Multiplica cada valor de entrada por el peso que conecta
         * esa entrada con la neurona de salida actual y lo acumula.
         *
         * El índice del peso es o*in_feat + i porque los pesos están
         * organizados por neurona de salida en memoria:
         *   [neurona 0: peso_0, peso_1, ..., peso_in]
         *   [neurona 1: peso_0, peso_1, ..., peso_in]
         *   ...
         */
        for (int i = 0; i < in_feat; i++)
            sum += x[b * in_feat + i] * weight[o * in_feat + i];

        /* Escribe el resultado en la posición de salida correspondiente */
        output[b * out_feat + o] = sum;
    }
}


/* =============================================================
 * RELU PARA VECTORES PLANOS
 *
 * Aplica max(0, x) a cada elemento del vector.
 * Se usa entre capas FC cuando se apilan varias seguidas para
 * introducir no linealidad entre ellas.
 *
 * Sin activaciones entre capas FC, apilar varias capas seguidas
 * sería equivalente a tener una sola capa, porque la composición
 * de funciones lineales sigue siendo lineal.
 * ============================================================= */
void relu_vector(float *v, int size) {
    for (int i = 0; i < size; i++)
        if (v[i] < 0.0f) v[i] = 0.0f;
}


/* =============================================================
 * SOFTMAX
 *
 * Convierte los logits de salida de la última capa FC en
 * probabilidades que suman exactamente 1.
 *
 * Fórmula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 *
 * Se resta el valor máximo antes de calcular los exponenciales
 * para evitar desbordamiento numérico (overflow). Restar una
 * constante a todos los elementos no cambia el resultado final
 * porque se cancela en el numerador y denominador, pero evita
 * que exp() reciba valores muy grandes y produzca infinito.
 * ============================================================= */
void softmax(float *v, int size) {

    /* Encuentra el valor máximo para estabilidad numérica */
    float max_val = v[0];
    for (int i = 1; i < size; i++)
        if (v[i] > max_val) max_val = v[i];

    /* Calcula exponenciales restando el máximo y acumula la suma */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        v[i] = expf(v[i] - max_val);
        sum += v[i];
    }

    /* Normaliza dividiendo cada elemento por la suma total */
    for (int i = 0; i < size; i++)
        v[i] /= sum;
}


/* =============================================================
 * DEMO
 *
 * Ejemplo con dos capas FC apiladas:
 *   Entrada 128 -> FC -> ReLU -> 64 -> FC -> Softmax -> 3 clases
 *
 * Los pesos se inicializan a valores fijos solo para probar.
 * En una red real vendrían de un archivo entrenado.
 * ============================================================= */
int main(void) {
    printf("=== Demo Fully Connected ===\n\n");

    int B       = 1;    /* una imagen                    */
    int in_feat = 128;  /* 128 características de entrada */
    int hidden  = 64;   /* 64 neuronas en la capa oculta  */
    int n_class = 3;    /* 3 clases de salida             */

    /* Vector de entrada con valores de prueba */
    float *x = (float*)malloc(B * in_feat * sizeof(float));
    for (int i = 0; i < in_feat; i++) x[i] = (float)i * 0.01f;

    /* Pesos y bias de la capa 1: conecta in_feat -> hidden */
    float *w1 = (float*)malloc(hidden * in_feat * sizeof(float));
    float *b1 = (float*)calloc(hidden, sizeof(float));
    for (int i = 0; i < hidden * in_feat; i++) w1[i] = 0.01f;

    /* Salida de la capa 1 */
    float *out1 = (float*)malloc(B * hidden * sizeof(float));
    fully_connected(x, w1, b1, out1, B, in_feat, hidden);
    relu_vector(out1, hidden);

    printf("Capa FC 1 + ReLU:\n");
    printf("  Entrada: [%d, %d]\n", B, in_feat);
    printf("  Salida:  [%d, %d]\n\n", B, hidden);

    /* Pesos y bias de la capa 2: conecta hidden -> n_class */
    float *w2 = (float*)malloc(n_class * hidden * sizeof(float));
    float *b2 = (float*)calloc(n_class, sizeof(float));
    for (int i = 0; i < n_class * hidden; i++) w2[i] = 0.02f;

    /* Salida de la capa 2 */
    float *out2 = (float*)malloc(B * n_class * sizeof(float));
    fully_connected(out1, w2, b2, out2, B, hidden, n_class);
    softmax(out2, n_class);

    printf("Capa FC 2 + Softmax:\n");
    printf("  Entrada: [%d, %d]\n", B, hidden);
    printf("  Salida:  [%d, %d]  (probabilidades por clase)\n\n",
           B, n_class);

    printf("Probabilidades:\n");
    for (int c = 0; c < n_class; c++)
        printf("  Clase %d: %.4f\n", c, out2[c]);

    free(x);
    free(w1); free(b1); free(out1);
    free(w2); free(b2); free(out2);

    printf("\n=== Fin ===\n");
    return 0;
}
