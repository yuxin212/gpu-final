#include <cstdlib>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>
#include <cstdio>
#include <time.h>

const static float lr = 1.0E-01f;

class Layer {
public:
  int m, n, o;
  float *output;
  float *preact;
  float *bias;
  float *weight;
  float *d_output;
  float *d_preact;
  float *d_weight;

  Layer(int m_, int n_, int o_);
  ~Layer();
  void setOutput(float *data);
  void clear();
  void bp_clear();
};

static Layer l_input = Layer(0, 0, 28 * 28);
static Layer l_c1 = Layer(5 * 5, 6, 24 * 24 * 6);
static Layer l_s1 = Layer(4 * 4, 1, 6 * 6 * 6);
static Layer l_f = Layer(6 * 6 * 6, 10, 10);

Layer::Layer(int m_, int n_, int o_) {
  m = m_;
  n = n_;
  o = o_;
  float h_bias[n];
  float h_weight[n][m];
  output = NULL;
  preact = NULL;
  bias = NULL;
  weight = NULL;
  for (int i = 0; i < n; i++) {
    h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
    for (int j = 0; j < m; j++) {
      h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
    }
  }
  cudaMalloc(&output, sizeof(float) * o);
  cudaMalloc(&preact, sizeof(float) * o);
  cudaMalloc(&bias, sizeof(float) * n);
  cudaMalloc(&weight, sizeof(float) * m * n);
  cudaMalloc(&d_output, sizeof(float) * o);
  cudaMalloc(&d_preact, sizeof(float) * o);
  cudaMalloc(&d_weight, sizeof(float) * m * n);
  cudaMemcpy(bias, h_bias, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(weight, h_weight, sizeof(float) * m * n, cudaMemcpyHostToDevice);
}

Layer::~Layer() {
  cudaFree(output);
  cudaFree(preact);
  cudaFree(bias);
  cudaFree(weight);
  cudaFree(d_output);
  cudaFree(d_preact);
  cudaFree(d_weight);
}

void Layer::setOutput(float *data) {
  cudaMemcpy(output, data, sizeof(float) * o, cudaMemcpyHostToDevice);
}

void Layer::clear() {
  cudaMemset(output, 0x00, sizeof(float) * o);
  cudaMemset(preact, 0x00, sizeof(float) * o);
}

void Layer::bp_clear() {
  cudaMemset(d_output, 0x00, sizeof(float) * o);
  cudaMemset(d_preact, 0x00, sizeof(float) * o);
  cudaMemset(d_weight, 0x00, sizeof(float) * m * n);
}

__device__ float sigmoid(float v) {
  return 1 / (1 + exp(-v));
}

__global__ void calc_sigmoid(float *input, float *output, const int N) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  for (int idx = N * pos / size; idx < N * (pos + 1) / size; idx++) {
    output[idx] = sigmoid(input[idx]);
  }
}

__global__ void calc_error(float *err, float *output, unsigned int y, const int N) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  for (int idx = N * pos / size; idx < N * (pos + 1) / size; idx++) {
    err[idx] = ((y == idx ? 1.0f : 0.0f) - output[idx]);
  }
}

__global__ void calc_grad(float *output, float *grad, const int N) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  for (int idx = N * pos / size; idx < N * (pos + 1) / size; idx++) {
    output[idx] += lr * grad[idx];
  }
}

__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 5 * 5 * 6 * 24 * 24;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 5);
    const int i2 = ((idx /= 5) % 5);
    const int i3 = ((idx /= 5) % 6);
    const int i4 = ((idx /= 6) % 24);
    const int i5 = ((idx /= 24) % 24);
    atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
  }
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6 * 24 * 24;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 6);
    const int i2 = ((idx /= 6) % 24);
    const int i3 = ((idx /= 24) % 24);
    preact[i1][i2][i3] += bias[i1];
  }
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 4 * 4 * 6 * 6 * 6;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 4);
    const int i2 = ((idx /= 4) % 4);
    const int i3 = ((idx /= 4) % 6);
    const int i4 = ((idx /= 6) % 6);
    const int i5 = ((idx /= 6) % 6);
    atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
  }
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6 * 6 * 6;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 6);
    const int i2 = ((idx /= 6) % 6);
    const int i3 = ((idx /= 6) % 6);
    preact[i1][i2][i3] += bias[0];
  }
}

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10 * 6 * 6 * 6;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 10);
    const int i2 = ((idx /= 10) % 6);
    const int i3 = ((idx /= 6) % 6);
    const int i4 = ((idx /= 6) % 6);
    atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
  }
}

__global__ void fp_bias_f(float preact[10], float bias[10]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10;
  for (int idx = N * pos / size; idx < N * (pos + 1) / size; idx++) {
    preact[idx] += bias[idx];
  }
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10 * 6 * 6 * 6;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 10);
    const int i2 = ((idx /= 10) % 6);
    const int i3 = ((idx /= 6) % 6);
    const int i4 = ((idx /= 6) % 6);
    d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
  }
}

__global__ void bp_bias_f(float bias[10], float d_preact[10]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10;
  for (int idx = N * pos / size; idx < N * (pos + 1) / size; idx++) {
    bias[idx] += lr * d_preact[idx];
  }
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 10 * 6 * 6 * 6;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 10);
    const int i2 = ((idx /= 10) % 6);
    const int i3 = ((idx /= 6) % 6);
    const int i4 = ((idx /= 6) % 6);
    atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
  }
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6 * 6 * 6;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 6);
    const int i2 = ((idx /= 6) % 6);
    const int i3 = ((idx /= 6) % 6);
    const float o = sigmoid(preact[i1][i2][i3]);
    d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
  }
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 1 * 4 * 4 * 6 * 6 * 6;
  const float d = pow(6.0f, 3.0f);
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 1);
    const int i2 = ((idx /= 1) % 4);
    const int i3 = ((idx /= 4) % 4);
    const int i4 = ((idx /= 4) % 6);
    const int i5 = ((idx /= 6) % 6);
    const int i6 = ((idx /= 6) % 6);
    atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
  }
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6 * 6 * 6;
  const float d = pow(6.0f, 3.0f);
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 6);
    const int i2 = ((idx /= 6) % 6);
    const int i3 = ((idx /= 6) % 6);
    atomicAdd(&bias[0], lr * d_preact[i1][i2][i3] / d);
  }
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 1 * 4 * 4 * 6 * 6 * 6;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 1);
    const int i2 = ((idx /= 1) % 4);
    const int i3 = ((idx /= 4) % 4);
    const int i4 = ((idx /= 4) % 6);
    const int i5 = ((idx /= 6) % 6);
    const int i6 = ((idx /= 6) % 6);
    atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
  }
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6 * 24 * 24;
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 6);
    const int i2 = ((idx /= 6) % 24);
    const int i3 = ((idx /= 24) % 24);
    const float o = sigmoid(preact[i1][i2][i3]);
    d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
  }
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6 * 5 * 5 * 24 * 24;
  const float d = pow(24.0f, 2.0f);
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 6);
    const int i2 = ((idx /= 6) % 5);
    const int i3 = ((idx /= 5) % 5);
    const int i4 = ((idx /= 5) % 24);
    const int i5 = ((idx /= 24) % 24);
    atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
  }
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blockDim.x * gridDim.x;
  const int N = 6 * 24 * 24;
  const float d = pow(24.0f, 2.0f);
  for (int n = N * pos / size; n < N * (pos + 1) / size; n++) {
    int idx = n;
    const int i1 = ((idx /= 1) % 6);
    const int i2 = ((idx /= 6) % 24);
    const int i3 = ((idx /= 24) % 24);
    atomicAdd(&bias[i1], lr * d_preact[i1][i2][i3] / d);
  }
}

static unsigned int bin2int(char *v) {
  int i;
  unsigned int ret = 0;
  for (i = 0; i < 4; i++) {
    ret <<= 8;
    ret |= (unsigned char)v[i];
  }
  return ret;
}

typedef struct mnist {
  double data[28][28];
  unsigned int label;
}mnist;

static mnist *train_data, *test_data;
static unsigned int train_size, test_size;

int load_mnist(const char *image, const char* label, mnist **data, unsigned int *count) {
  int return_code = 0;
  int i;
  char tmp[4];
  unsigned int image_size, label_size;
  unsigned int image_dim[2];
  FILE *ifp = fopen(image, "rb");
  FILE *lfp = fopen(label, "rb");
  if (!ifp || !lfp) {
    return_code = -1;
    if (ifp) fclose(ifp);
    if (lfp) fclose(lfp);
    return return_code;
  }
  fread(tmp, 1, 4, ifp);
  if (bin2int(tmp) != 2051) {
    return_code = -2;
    if (ifp) fclose(ifp);
    if (lfp) fclose(lfp);
    return return_code;
  }
  fread(tmp, 1, 4, lfp);
  if (bin2int(tmp) != 2049) {
    return_code = -3;
    if (ifp) fclose(ifp);
    if (lfp) fclose(lfp);
    return return_code;
  }
  fread(tmp, 1, 4, ifp);
  image_size = bin2int(tmp);
  fread(tmp, 1, 4, lfp);
  label_size = bin2int(tmp);
  if (image_size != label_size) {
    return_code = -4;
    if (ifp) fclose(ifp);
    if (lfp) fclose(lfp);
    return return_code;
  }
  for (i = 0; i < 2; i++) {
    fread(tmp, 1, 4, ifp);
    image_dim[i] = bin2int(tmp);
  }
  if (image_dim[0] != 28 || image_dim[1] != 28) {
    return_code = -2;
    if (ifp) fclose(ifp);
    if (lfp) fclose(lfp);
    return return_code;
  }
  *count = image_size;
  *data = (mnist *)malloc(sizeof(mnist) * image_size);
  for (i = 0; i < image_size; i++) {
    int j;
    unsigned char read_data[28 * 28];
    mnist *d = &(*data)[i];
    fread(read_data, 1, 28 * 28, ifp);
    for (j = 0; j < 28 * 28; j++) {
      d->data[j / 28][j % 28] = read_data[j] / 255.0;
    }
    fread(tmp, 1, 1, lfp);
    d->label = tmp[0];
  }
  if (ifp) fclose(ifp);
  if (lfp) fclose(lfp);
  return return_code;
}

static double forward_prop(double data[28][28]) {
  float input[28][28];
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      input[i][j] = data[i][j];
    }
  }
  l_input.clear();
  l_c1.clear();
  l_s1.clear();
  l_f.clear();
  clock_t start, end;
  start = clock();
  l_input.setOutput((float *)input);
  fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float(*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
  fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
  calc_sigmoid<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.o);
  fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
  fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
  calc_sigmoid<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.o);
  fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
  fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
  calc_sigmoid<<<64, 64>>>(l_f.preact, l_f.output, l_f.o);
  end = clock();
  return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static double back_prop() {
  clock_t start, end;
  start = clock();
  bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
  bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);
  bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
  bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
  bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
  bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
  bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
  bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
  bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
  bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
  calc_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.m * l_f.n);
  calc_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.m * l_s1.n);
  calc_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.m * l_c1.n);
  end = clock();
  return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static unsigned int classify(double data[28][28]) {
  float output[10];
  forward_prop(data);
  unsigned int max = 0;
  cudaMemcpy(output, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
  for (int i = 1; i < 10; i++) {
    if (output[max] < output[i]) {
      max = i;
    }
  }
  return max;
}

static void test() {
  int error = 0;
  fprintf(stdout, "Evaluating on the test set...\n");
  for (int i = 0; i < test_size; i++) {
    if (classify(test_data[i].data) != test_data[i].label) {
      error++;
    }
  }
  printf("Test Error: %.2lf%%\n", double(error) / double(test_size) * 100.0);
}

static void learn() {
  static cublasHandle_t blas;
  cublasCreate(&blas);
  float err;
  int iter = 10;
  double total_time = 0.0;
  printf("Start Training...\n");
  for (int i = 0; i < iter; i++) {
  //while (iter < 0 || iter-- > 0) {
    err = 0.0f;
    for (int j = 0; j < train_size; j++) {
      float temp_err;
      time_taken += forward_prop(train_data[j].data);
      l_f.bp_clear();
      l_s1.bp_clear();
      l_c1.bp_clear();
      calc_error<<<10, 1>>>(l_f.d_preact, l_f.output, train_data[j].label, 10);
      cublasSnrm2(blas, 10, l_f.d_preact, 1, &temp_err);
      err += temp_err;
      total_time += back_prop();
    }
    err /= train_size;
    printf("Epoch %d/10, training error: %e, total training time: %lf s.\n", i + 1, err, total_time);
  }
  printf("\n Total training time: %lf s.\n", total_time);
}

static void loaddata() {
  fprintf(stdout, "Start loading MNIST data...\n");
  int r1 = load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", &train_data, &train_size);
  int r2 = load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", &test_data, &test_size);
  if (r1 != 0) {
    printf("Failed to load training dataset. Now exiting. \n");
    exit(-1);
  }
  if (r2 != 0) {
    printf("Failed to load test dataset. Now exiting. \n");
    exit(-1);
  }
  printf("Size of training data: %d\n", train_size);
  printf("Size of test data: %d\n", test_size);
}

static void individual_test(int i) {
  printf("Number %d data in test dataset.\n", i);
  printf("Classification result: %d. Ground truth: %d\n\n", classify(test_data[i].data), test_data[i].label);
}

int main() {
  srand(time(NULL));
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    printf("CUDA initialization failed with error code %d\n", err);
    return 1;
  }
  loaddata();
  learn();
  test();
  printf("Start randomly select 5 data from test dataset to test.\n");
  for (int i = 0; i < 5; i++) {
    int j = rand() % 10000;
    individual_test(j);
  }
  return 0;
}
