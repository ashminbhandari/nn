#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_FEATURES 784
#define NN_OUTPUT 10

typedef struct Layer Layer;
typedef struct Network Network;

struct Network
{
    int num_layers;
    Layer *layers;
};

struct Layer
{
    int input_size;
    int output_size;
    float *biases;
    float *weights;
    float *outputs;
};

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

void softmax(float *inputs, float *outputs, int size)
{
    float max_val = inputs[0];
    for (int i = 1; i < size; i++)
    {
        if (inputs[i] > max_val)
            max_val = inputs[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        outputs[i] = expf(inputs[i] - max_val);
        sum += outputs[i];
    }
    for (int i = 0; i < size; i++)
    {
        outputs[i] /= sum;
    }
}

float cross_entropy_loss(float *truth, float *predicted)
{
    float loss = 0.0f, epsilon = 1e-9;
    for (int i = 0; i < NN_OUTPUT; i++)
    {
        loss += truth[i] * logf(predicted[i] + epsilon);
    }
    return -loss;
}

void forward_pass(struct Network *n, float *outputs)
{
    float *nn_inputs = malloc(N_FEATURES * sizeof(float));
    for (int i = 0; i < N_FEATURES; i++)
    {
        nn_inputs[i] = 1.0;
    }

    // Layer 1
    for (int i = 0; i < n->layers[0].output_size; i++)
    {
        n->layers[0].outputs[i] = n->layers[0].biases[i];
        for (int j = 0; j < N_FEATURES; j++)
        {
            n->layers[0].outputs[i] += n->layers[0].weights[i * N_FEATURES + j] * nn_inputs[j];
        }
        n->layers[0].outputs[i] = sigmoid(n->layers[0].outputs[i]);
    }

    // Layer 2
    for (int i = 0; i < n->layers[1].output_size; i++)
    {
        n->layers[1].outputs[i] = n->layers[1].biases[i];
        for (int j = 0; j < n->layers[1].input_size; j++)
        {
            n->layers[1].outputs[i] += n->layers[1].weights[i * n->layers[1].input_size + j] * n->layers[0].outputs[j];
        }
    }
    softmax(n->layers[1].outputs, outputs, NN_OUTPUT);

    free(nn_inputs);
}

int main()
{
    Layer l1 = {
        .input_size = 784,
        .output_size = 128,
        .biases = calloc(128, sizeof(float)),
        .weights = malloc(784 * 128 * sizeof(float)),
        .outputs = calloc(128, sizeof(float))};
    Layer l2 = {
        .input_size = 128,
        .output_size = 10,
        .biases = calloc(10, sizeof(float)),
        .weights = malloc(128 * 10 * sizeof(float)),
        .outputs = calloc(10, sizeof(float))};

    for (int i = 0; i < 784 * 128; i++)
        l1.weights[i] = 0.01;
    for (int i = 0; i < 128 * 10; i++)
        l2.weights[i] = 0.01;

    float truth[10] = {0, 0, 0.01, 0.001, 1, 0, 0, 0, 0, 0};
    float *nn_outputs = malloc(NN_OUTPUT * sizeof(float));

    Network n = {2, (Layer[]){l1, l2}};
    forward_pass(&n, nn_outputs);

    float loss = cross_entropy_loss(truth, nn_outputs);
    printf("Loss: %f\n", loss);
    for (int i = 0; i < 10; i++)
    {
        printf("Output %d: %f\n", i, nn_outputs[i]);
    }

    free(l1.weights);
    free(l1.biases);
    free(l1.outputs);
    free(l2.weights);
    free(l2.biases);
    free(l2.outputs);
    free(nn_outputs);
    return 0;
}