#include "cudaResize.h"

// gpuResample
__global__ void gpuResizeSimple(float2 scale, int *input, int iWidth, int *output, int oWidth, int oHeight) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oWidth || y >= oHeight)
    return;

  const int dx = (x * scale.x);
  const int dy = (y * scale.y);

  const int px = input[dy * iWidth + dx];

  output[y * oWidth + x] = px;
}


__global__ void gpuResizeAverage(float2 scale, int *input, int iWidth, int *output, int oWidth, int oHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oWidth || y >= oHeight) {
    return;
  }

  int dx = (x * scale.x);
  int dy = (y * scale.y);

  int r = 0, g = 0, b = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      r += input[(dy + i) * iWidth + (dx + j)] & 0xff;
      g += (input[(dy + i) * iWidth + (dx + j)] >> 8) & 0xff;
      b += (input[(dy + i) * iWidth + (dx + j)] >> 16) & 0xff;
    }
  }

  output[y * oWidth + x] = (r / 9) + ((g / 9) << 8) + ((b / 9) << 16);
}

__global__ void gpuResizeBilinear(float2 scale, int *input, int iWidth, int *output, int oWidth, int oHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oWidth || y >= oHeight) {
    return;
  }

  int dx = (x * scale.x);
  int dy = (y * scale.y);
  double center_x = dx + scale.x / 2;
  double center_y = dy + scale.y / 2;
  int window[4][3];
  window[0][0] = input[(int) (center_y - scale.y / 2) * iWidth + (int) (center_x - scale.x / 2)] & 0xff;
  window[0][1] = (input[(int) (center_y - scale.y / 2) * iWidth + (int) (center_x - scale.x / 2)] >> 8) & 0xff;
  window[0][2] = (input[(int) (center_y - scale.y / 2) * iWidth + (int) (center_x - scale.x / 2)] >> 16) & 0xff;

  window[1][0] = input[(int) (center_y + scale.y / 2) * iWidth + (int) (center_x - scale.x / 2)] & 0xff;
  window[1][1] = (input[(int) (center_y + scale.y / 2) * iWidth + (int) (center_x - scale.x / 2)] >> 8) & 0xff;
  window[1][2] = (input[(int) (center_y + scale.y / 2) * iWidth + (int) (center_x - scale.x / 2)] >> 16) & 0xff;

  window[2][0] = input[(int) (center_y - scale.y / 2) * iWidth + (int) (center_x + scale.x / 2)] & 0xff;
  window[2][1] = (input[(int) (center_y - scale.y / 2) * iWidth + (int) (center_x + scale.x / 2)] >> 8) & 0xff;
  window[2][2] = (input[(int) (center_y - scale.y / 2) * iWidth + (int) (center_x + scale.x / 2)] >> 16) & 0xff;

  window[3][0] = input[(int) (center_y + scale.y / 2) * iWidth + (int) (center_x + scale.x / 2)] & 0xff;
  window[3][1] = (input[(int) (center_y + scale.y / 2) * iWidth + (int) (center_x + scale.x / 2)] >> 8) & 0xff;
  window[3][2] = (input[(int) (center_y + scale.y / 2) * iWidth + (int) (center_x + scale.x / 2)] >> 16) & 0xff;

  int finalBytes[3];
  for (int i = 0; i < 3; ++i) {
    double x_axis_interpolation_lower = ((int) (center_x + scale.x / 2) - center_x) * window[0][i] /
                                        ((int) (center_x + scale.x / 2) - (int) (center_x - scale.x / 2)) +
                                        (center_x - (int) (center_x - scale.x / 2)) * window[2][i] /
                                        ((int) (center_x + scale.x / 2) - (int) (center_x - scale.x / 2));
    double x_axis_interpolation_higher = ((int) (center_x + scale.x / 2) - center_x) * window[1][i] /
                                         (int(center_x + scale.x / 2) - (int) (center_x - scale.x / 2)) +
                                         (center_x - (int) (center_x - scale.x / 2)) * window[3][i] /
                                         ((int) (center_x + scale.x / 2) - (int) (center_x - scale.x / 2));

    finalBytes[i] =
            ((int) (center_y + scale.y / 2) - center_y) * x_axis_interpolation_lower /
            ((int) (center_y + scale.y / 2) - (int) (center_y - scale.y / 2)) +
            (center_y - (int) (center_y - scale.y / 2)) * x_axis_interpolation_higher /
            ((int) (center_y + scale.y / 2) - (int) (center_y - scale.y / 2));
  }

  output[y * oWidth + x] = finalBytes[0] + (finalBytes[1] << 8) + (finalBytes[2] << 16);
}

// cudaResize
void cudaResize(int *input, size_t inputWidth, size_t inputHeight,
                int *output, size_t outputWidth, size_t outputHeight) {
  if (!input || !output)
    return;

  if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0)
    return;

  const float2 scale = make_float2(float(inputWidth) / float(outputWidth),
                                   float(inputHeight) / float(outputHeight));

  // launch kernel
  const dim3 blockDim(8, 8);
//  const dim3 gridDim(outputWidth / blockDim.x, outputHeight / blockDim.y);
  const dim3 gridDim((outputWidth - 1 + blockDim.x - 1) / blockDim.x,
                     (outputHeight - 1 + (blockDim.x - 1)) / blockDim.y);

  int *cudaNewPic, *cudaArrayPic;
  cudaMalloc((void **) &cudaNewPic, outputWidth * outputHeight * sizeof(int));
  cudaMalloc((void **) &cudaArrayPic, inputWidth * inputHeight * sizeof(int));

  cudaMemcpy(cudaArrayPic, input, inputWidth * inputHeight * sizeof(int), cudaMemcpyHostToDevice);

  //gpuResize<<< gridDim, blockDim >>>(scale, cudaArrayPic, inputWidth, cudaNewPic, outputWidth, outputHeight);
  //gpuResizeAverage<<< gridDim, blockDim >>>(scale, cudaArrayPic, inputWidth, cudaNewPic, outputWidth, outputHeight);
  gpuResizeBilinear<<< gridDim, blockDim >>>(scale, cudaArrayPic, inputWidth, cudaNewPic, outputWidth, outputHeight);

  cudaMemcpy(output, cudaNewPic, outputWidth * outputHeight * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(cudaNewPic);
  cudaFree(cudaArrayPic);
}
