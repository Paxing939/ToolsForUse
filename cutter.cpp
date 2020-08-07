#include <iostream>
#include <list>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// обработчик событий от мышки
const int alpha_slider_max = 100;
int alpha_slider = 25;
cv::Mat colored_frame;
cv::Point center;
int h = 25, w = 25;
std::vector<cv::Rect> rectangles;

void on_trackbar(int, void *) {
    if (!rectangles.empty()) {
        auto tmp = cv::Rect(*(rectangles.end() - 1));
        rectangles.erase(rectangles.end() - 1);
        int x = tmp.x + w / 2, y = tmp.y + h / 2;
        w = h = alpha_slider;
        rectangles.emplace_back(x - w / 2, y - h / 2, w, h);
    }
    auto tmp = colored_frame.clone();
    for (const auto &rectangle : rectangles) {
        cv::rectangle(tmp, rectangle, cv::Scalar(255, 0, 0), 1, 8, 0);
    }
    cv::imshow("result", tmp);
}

static void onMouse(int event, int x, int y, int, void *) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;
    auto tmp = colored_frame.clone();
    center = {x, y};
    cv::Rect new_rectangle(x - w / 2, y - h / 2, w, h);
    if (!rectangles.empty() && x >= rectangles.back().x && y >= rectangles.back().y && x <= rectangles.back().x + w &&
        y <= rectangles.back().y + h) {
        rectangles.erase(rectangles.end() - 1);
    }
    rectangles.push_back(new_rectangle);
    std::cerr << new_rectangle << '\n';
    for (const auto &rectangle : rectangles) {
        cv::rectangle(tmp, rectangle, cv::Scalar(255, 0, 0), 1, 8, 0);
    }
    imshow("result", tmp);
}

int CutSomePics() {
    cv::VideoCapture cap("/home/ilya/source_data/2020_05_19__15_25_03.avi");
    cv::Mat previous_frame;
//  for (int i = 0; i < 75; ++i) {
//    cap >> previous_frame;
//  }

    cv::Mat frame = cv::imread("/home/ilya/f/finp" + std::to_string(0) + ".jpg");
    colored_frame = frame;
    cv::namedWindow("result");
    cv::setMouseCallback("result", onMouse, nullptr);
    cv::createTrackbar("Track bar", "result", &alpha_slider, alpha_slider_max, on_trackbar);
    int i = 0;
    while (cap.isOpened()) {
        cap >> frame;
//  for (int i = 0; i < 135; ++i) {
//    frame = cv::imread("/home/ilya/f/finp" + std::to_string(i) + ".jpg");
        while (true) {
            colored_frame = frame;
            cv::imshow("result", colored_frame);
            int c = cv::waitKey();
            if (c == 27 || c == 32) {
                break;
            }
        }

        int j = 0;
        for (const auto &rectangle : rectangles) {
            cv::imwrite(
                    "/home/ilya/source_data/Data_for_YOLO/video_from_PARAD2/Папки/2020_05_19__15_25_03.avi birds меньше чем 25х25/" +
                    std::to_string(i++) + "_" + std::to_string(j++) + ".bmp", frame(rectangle));
        }

        rectangles.clear();
    }
    return 0;
}