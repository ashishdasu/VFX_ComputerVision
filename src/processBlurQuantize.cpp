// Ashish Dasu
// Process one image with blur+quantize for report

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [levels]" << std::endl;
        std::cout << "  levels: quantization levels (default: 10)" << std::endl;
        return -1;
    }

    // Read input image
    cv::Mat src = cv::imread(argv[1]);
    if (src.data == NULL) {
        std::cerr << "Unable to read image: " << argv[1] << std::endl;
        return -1;
    }

    // Get quantization levels
    int levels = 10;
    if (argc >= 3) {
        levels = atoi(argv[2]);
    }

    std::cout << "Processing image: " << argv[1] << std::endl;
    std::cout << "Size: " << src.cols << "x" << src.rows << std::endl;
    std::cout << "Quantization levels: " << levels << std::endl;

    // Apply blur+quantize
    cv::Mat blurQuant;
    blurQuantize(src, blurQuant, levels);
    cv::imwrite("../report/blur_quantize.png", blurQuant);
    std::cout << "Saved: blur_quantize.png" << std::endl;

    cv::imwrite("../report/blur_quantize_original.png", src);
    std::cout << "Saved: blur_quantize_original.png (copy of input)" << std::endl;

    std::cout << "\nBlur+quantize images saved to report/ folder!" << std::endl;
    return 0;
}
