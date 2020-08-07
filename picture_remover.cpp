#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>

#include "functions.h"

using namespace std;
using namespace cv;

int RemoveSomePics() {
    string folder = "/home/ilya/Downloads/";
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                std::string filename =
                        "Рессора_p="s + to_string(i) + "_d="s + to_string(j) + "_q="s + to_string(k) + ""s + ""s +
                        "_fig.png"s;
                std::cerr << filename;
                auto image = imread(folder + filename);
                if (image.empty()) {
                    continue;
                }
                imshow(filename, image);
                int c = waitKey();
                if (c == 255) {
                    remove(filename.c_str());
                }
            }
        }
    }


    std::cout << "Hello, World!" << std::endl;
    return 0;
}
