#include "cudaResize.h"
#include "log_duration.h"
#include "functions.h"

int *ReduceImageUsingSplines(const int *const image, int width, int height, double window_size) {
  int *result = new int[(int) ((double) (width * height) / (window_size * window_size))];

  for (int i = 2; i < height - 2; ++i) {
    for (int j = 2; j < width - 2; ++j) {
      int c[16], b[16];
      for (int k = -2; k < 2; ++k) {
        for (int p = -2; p < 2; ++p) {
          int pixel = image[int(i + window_size * k) * width + int(j + window_size * p)];
          b[(k + 2) * 4 + (p + 2)] = pixel;
        }
      }

      int a[16][16] = {{0,  0,   0,   0,  0,   36,  0,   0,  0,   0,       0,     0,  0,  0,  0,  0},
                       {0,  -12, 0,   0,  0,   -18, 0,   0,  0,   36,      0,     0,  0,  -6, 0,  0},
                       {0,  18,  0,   0,  0,   -36, 0,   0,  0,   18,      0,     0,  0,  0,  0,  0},
                       {0,  -6,  0,   0,  0,   18,  0,   0,  0,   -18,     0,     0,  0,  6,  0,  0},
                       {0,  0,   0,   0,  -12, -18, 36,  -6, 0,   0,       0,     0,  0,  0,  0,  0},
                       {4,  6,   -12, 2,  6,   9,   -18, 3,  -12, -18,     36,    -6, 2,  3,  -6, 1},
                       {-6, -9,  18,  -3, 12,  18,  36,  6,  -6,  -6,      18,    3,  0,  0,  0,  0},
                       {2,  3,   -6,  1,  -6,  -9,  18,  -3, 6,   9,       -18,   3,  -2, -3, 6,  -1},
                       {0,  0,   0,   0,  18,  -36, 18,  0,  0,   0,       0,     0,  0,  0,  0,  0},
                       {-6, 12,  -6,  0,  -9,  18,  -9,  0,  -18, 36,      18,    0,  -3, 6,  -3, 0},
                       {9,  -18, 9,   0,  -18, 36,  -18, 0,  9,   -18,     9,     0,  0,  0,  0,  0},
                       {-3, 6,   -3,  0,  9,   -18, 9,   0,  -9,  18,      -9,    0,  3,  -6, 3,  0},
                       {0,  0,   0,   0,  -6,  18,  -18, -6, 0,   0,       0,     0,  0,  0,  0,  0},
                       {2,  -6,  6,   -2, 3,   -9,  9,   -3, -6,  18 - 18, 6,     1,  -3, 3,  -1},
                       {-3, 9,   -9,  3,  6,   -18, 18,  -6, -3,  9,       -9,    3,  0,  0,  0,  0},
                       {1,  -3,  3,   -1, -3,  9,   -9,  3,  3,   -9,      9 - 3, -1, 3,  -3, 1},};

      int A = 16, B = 16;
      for (int i = 0; i < A; i++) {
        for (int k = 0; k < B; k++) {
          c[i] += a[i][k] * b[k];
        }
      }

      result[(int) (i * width / window_size + height / window_size)] =
              c[0] * b[5] + c[1] * b[9] + c[2] * b[6] + c[3] * b[10] +
              c[4] * b[1] + c[5] * b[4] + c[6] * b[2] + c[7] * b[8] +
              c[8] * b[13] + c[9] * b[7] + c[10] * b[0] + c[11] * b[14] +
              c[12] * b[11] + c[13] * b[12] + c[14] * b[3] + c[15] * b[14];
    }
  }
}

std::vector<double> bicubicresize(const std::vector<int> &in,
                                  std::size_t src_width, std::size_t src_height,
                                  std::size_t dest_width, std::size_t dest_height) {
  std::vector<double> out(dest_width * dest_height);

  const float tx = float(src_width) / dest_width;
  const float ty = float(src_height) / dest_height;
  const int components = 1;
  const int bytes_per_row = src_width * components;

  const int components2 = components;
  const int bytes_per_row2 = dest_width * components;

  double Cc;
  double C[5];
  double d0, d2, d3, a0, a1, a2, a3;

  for (int i = 0; i < dest_height; ++i) {
    for (int j = 0; j < dest_width; ++j) {
      const int x = int(tx * j);
      const int y = int(ty * i);
      const float dx = tx * j - x;
      const float dy = ty * i - y;

      for (int jj = 0; jj <= 3; ++jj) {
        d0 = in[(y - 1 + jj) * bytes_per_row + (x - 1) * components] -
             in[(y - 1 + jj) * bytes_per_row + (x) * components];
        d2 = in[(y - 1 + jj) * bytes_per_row + (x + 1) * components] -
             in[(y - 1 + jj) * bytes_per_row + (x) * components];
        d3 = in[(y - 1 + jj) * bytes_per_row + (x + 2) * components] -
             in[(y - 1 + jj) * bytes_per_row + (x) * components];
        a0 = in[(y - 1 + jj) * bytes_per_row + (x) * components];
        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
        a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
        a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
        C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

        d0 = C[0] - C[1];
        d2 = C[2] - C[1];
        d3 = C[3] - C[1];
        a0 = C[1];
        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
        a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
        a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
        Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
        out[i * bytes_per_row2 + j * components2] = Cc;
//        out[i * bytes_per_row2 + j * components2] /= 255.;
      }
    }
  }

  return out;
}

int w;
int channels = 1;

unsigned char get_subpixel(const std::vector<unsigned char> &bmap, int y, int x, int k) {
  return bmap[y * w * channels + x + k];
}

inline unsigned char saturate(float x) {
  return x > 255.0f ? 255 : x < 0.0f ? 0 : (unsigned char) x;
}

std::vector<unsigned char>
bicubic_resize(std::vector<unsigned char> &bmap, std::size_t bmap_width, std::size_t bmap_height,
               std::size_t channels, std::size_t dest_width, std::size_t dest_height) {
  w = bmap_width;
  std::vector<unsigned char> out(dest_width * dest_height * channels);

  const double tx = double(bmap_width) / dest_width;
  const double ty = double(bmap_height) / dest_height;
  const std::size_t row_stride = dest_width * channels;
  float C[5] = {0};

  for (unsigned i = 0; i < dest_height; ++i) {
    for (unsigned j = 0; j < dest_width; ++j) {
      const float x = float(tx * j);
      const float y = float(ty * i);
      const float dx = tx * j - x, dx2 = dx * dx, dx3 = dx2 * dx;
      const float dy = ty * i - y, dy2 = dy * dy, dy3 = dy2 * dy;

      for (int k = 0; k < 3; ++k) {
        for (int jj = 0; jj < 4; ++jj) {
          const int idx = y - 1 + jj;
          float a0 = get_subpixel(bmap, idx, x, k);
          float d0 = get_subpixel(bmap, idx, x - 1, k) - a0;
          float d2 = get_subpixel(bmap, idx, x + 1, k) - a0;
          float d3 = get_subpixel(bmap, idx, x + 2, k) - a0;
          float a1 = -(1.0f / 3.0f) * d0 + d2 - (1.0f / 6.0f) * d3;
          float a2 = 0.5f * d0 + 0.5f * d2;
          float a3 = -(1.0f / 6.0f) * d0 - 0.5f * d2 + (1.0f / 6.0f) * d3;
          C[jj] = a0 + a1 * dx + a2 * dx2 + a3 * dx3;

          d0 = C[0] - C[1];
          d2 = C[2] - C[1];
          d3 = C[3] - C[1];
          a0 = C[1];
          a1 = -(1.0f / 3.0f) * d0 + d2 - (1.0f / 6.0f) * d3;
          a2 = 0.5f * d0 + 0.5f * d2;
          a3 = -(1.0f / 6.0f) * d0 - 0.5f * d2 + (1.0f / 6.0f) * d3;
          out[i * row_stride + j * channels + k] = saturate(a0 + a1 * dy + a2 * dy2 + a3 * dy3);
        }
      }
    }
  }

  return out;
}

std::vector<unsigned char> MatToUChar(cv::Mat image) {
  std::vector<unsigned char> char_pointer_source(image.rows * image.cols * channels);
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
//      char_pointer_source[i * image.cols * channels + j] = image.at<cv::Vec3b>(i, j)[0];
//      char_pointer_source[i * image.cols * channels + j + 1] = image.at<cv::Vec3b>(i, j)[1];
//      char_pointer_source[i * image.cols * channels + j + 2] = image.at<cv::Vec3b>(i, j)[2];
      char_pointer_source[i * image.cols * channels + j] = image.at<uchar>(i, j);
    }
  }
  return char_pointer_source;
}

cv::Mat ImageToMat(const std::vector<unsigned char> &image, int width, int height) {
  cv::Mat result = cv::Mat(height, width, CV_8UC(channels));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
//      result.at<cv::Vec3b>(i, j) = {image[i * width * channels + j],
//                                    image[i * width * channels + j + 1],
//                                    image[i * width * channels + j + 2]};
      result.at<uchar>(i, j) = image[i * width * channels + j];
    }
  }
  return result;
}

//int main() {
//  int result_width = 1000, result_height = 1000;
//  cv::Mat source = cv::imread(
////          "/home/ilya/source_data/Data_for_YOLO/video_from_PARAD2/Папки/исходники для ресайза/2020_05_19__15_13_51.avi birds меньше чем 25х25/22_0.bmp");
//          "/home/ilya/Archive/1.bmp");
//  cv::cvtColor(source, source, cv::COLOR_BGR2GRAY);
//  channels = source.channels();
//  cv::imshow("source", source);
//  auto in = MatToUChar(source);
//  auto out = bicubic_resize(in, source.cols, source.rows, source.channels(), result_width, result_height);
//  auto result = ImageToMat(out, result_width, result_height);
//
//  cv::imwrite("/home/ilya/11212.bmp", result);
//  cv::imshow("custom", result);
//
//  cv::resize(source, source, cv::Size(result_width, result_height), 0, 0, cv::INTER_CUBIC);
//  cv::imshow("opencv", result);
//  cv::absdiff(source, result, result);
//  cv::imshow("diff", result);
//  cv::imwrite("/home/ilya/вшаа.bmp", result);
//  cv::waitKey();
//  return 0;
//}

int ResizeSomePics() {
  int refuse = 0, birds = 0, no = 0, l = 0;
  for (int i = 0; i < 10000; ++i) {
//    for (int j = 0; j < 40; j++) {
//    bool negative = false;
    cv::Mat image = cv::imread(
            "/home/ilya/source_data/Data_for_YOLO/video_from_PARAD2/unity (20 obj Frames) Third Camera redux/" +
            std::to_string(i) + /*"_" + std::to_string(j) +*/ ".bmp");


//    if (image.empty()) {
//      image = cv::imread("/home/ilya/Downloads/Telegram Desktop/TrainSamples/" + std::to_string(i) + "_no.bmp");
//      negative = true;
//    }

    if (image.empty()) {
//        refuse++;
      continue;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    if (image.cols == 25 && image.rows == 25) {
      cv::imwrite(
              "/home/ilya/source_data/Data_for_YOLO/video_from_PARAD2/unity (20 obj Frames) Third Camera redux 25х25/" +
              std::to_string(l++) + ".bmp", image);
      continue;
    }

    std::vector<int> pic(image.rows * image.cols);
    cv::Vec3b cCurrColorGetPict;
    {
      for (int i = 0, l = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          pic[l++] = image.at<cv::Vec3b>(i, j)[0] + (int(image.at<cv::Vec3b>(i, j)[1]) << 8) +
                     (int(image.at<cv::Vec3b>(i, j)[1]) << 16);
        }
      }
    }


    int result_width = 25, result_height = 25;
    auto in = MatToUChar(image);
    std::vector<unsigned char> out;
    {
//      LOG_DURATION("sgdg");
      out = bicubic_resize(in, image.cols, image.rows, image.channels(), result_width, result_height);
    }
    image = ImageToMat(out, result_width, result_height);

    cv::imwrite(
            "/home/ilya/source_data/Data_for_YOLO/video_from_PARAD2/unity (20 obj Frames) Third Camera redux 25х25/" +
            std::to_string(l++) + ".bmp", image);
    std::cerr << i << '\n';
//    }
  }

//  std::cerr << "Wrong pics: " << refuse << '\n';
  std::cerr << "Birds: " << birds << ", No: " << no;
}
