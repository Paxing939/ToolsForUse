#ifndef IMJPROC_LOG_DURATION_H
#define IMJPROC_LOG_DURATION_H

#include <chrono>
#include <string>
#include <utility>
#include <iostream>

#define ID_CONCAT_REALIZATION(x, y) x##y
#define ID_CONCAT(x, y) ID_CONCAT_REALIZATION(x, y)
#define UNIQUE_ID ID_CONCAT(___local_variable___, __LINE__)

#define LOG_DURATION(message) LogDuration UNIQUE_ID(message)

class LogDuration {
public:
    explicit LogDuration(std::string hint)
            : hint_(std::move(hint)), begin_(std::chrono::steady_clock::now()) {
    }

    ~LogDuration() {
        auto end = std::chrono::steady_clock::now();

        std::cerr << hint_ << ": "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin_).count()
                  << " microseconds" << '\n';
    }

private:
    std::string hint_;
    std::chrono::time_point<std::chrono::steady_clock> begin_;
};

#endif //IMJPROC_LOG_DURATION_H
