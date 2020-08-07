#pragma once

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

void cudaResize(int *input, size_t inputWidth, size_t inputHeight,
                int *output, size_t outputWidth, size_t outputHeight);
