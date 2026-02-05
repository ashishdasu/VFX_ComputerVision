// Ashish Dasu
// Process one image with Sobel X, Sobel Y, and magnitude for report

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Read input image
    cv::Mat src = cv::imread(argv[1]);
    if (src.data == NULL) {
        std::cerr << "Unable to read image: " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "Processing image: " << argv[1] << std::endl;
    std::cout << "Size: " << src.cols << "x" << src.rows << std::endl;

    // Apply Sobel X
    cv::Mat sobelX;
    sobelX3x3(src, sobelX);
    cv::Mat sobelX_display;
    cv::convertScaleAbs(sobelX, sobelX_display);
    cv::imwrite("../report/sobel_x.png", sobelX_display);
    std::cout << "Saved: sobel_x.png" << std::endl;

    // Apply Sobel Y
    cv::Mat sobelY;
    sobelY3x3(src, sobelY);
    cv::Mat sobelY_display;
    cv::convertScaleAbs(sobelY, sobelY_display);
    cv::imwrite("../report/sobel_y.png", sobelY_display);
    std::cout << "Saved: sobel_y.png" << std::endl;

    // Apply gradient magnitude
    cv::Mat mag;
    magnitude(sobelX, sobelY, mag);
    cv::imwrite("../report/sobel_magnitude.png", mag);
    std::cout << "Saved: sobel_magnitude.png" << std::endl;

    cv::imwrite("../report/sobel_original.png", src);
    std::cout << "Saved: sobel_original.png (copy of input)" << std::endl;

    std::cout << "\nAll Sobel images saved to report/\n" << std::endl;
    return 0;
}
