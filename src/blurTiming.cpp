/* Ashish Dasu
 * January 2026
 *
 * Program to test and compare the performance of two blur implementations.
 * Tests blur5x5_1 (naive) vs blur5x5_2 (optimized separable filters).
 *
 * Usage: ./blurTiming <image_path>
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "filters.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Read the input image
    cv::Mat src = cv::imread(argv[1]);
    if (src.empty()) {
        std::cerr << "Error: Could not read image from " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "Image size: " << src.cols << " x " << src.rows << std::endl;
    std::cout << "Running blur tests (averaging over 10 iterations)...\n" << std::endl;

    cv::Mat dst1, dst2;
    const int iterations = 10;

    // Test blur5x5_1 (naive implementation)
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        blur5x5_1(src, dst1);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    double avgTime1 = duration1.count() / (double)iterations / 1000.0;  // Convert to milliseconds

    // Test blur5x5_2 (optimized separable filter)
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        blur5x5_2(src, dst2);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    double avgTime2 = duration2.count() / (double)iterations / 1000.0;  // Convert to milliseconds

    // Display results
    std::cout << "=== TIMING RESULTS ===" << std::endl;
    std::cout << "blur5x5_1 (naive):     " << avgTime1 << " ms" << std::endl;
    std::cout << "blur5x5_2 (separable): " << avgTime2 << " ms" << std::endl;
    std::cout << "Speedup:               " << avgTime1 / avgTime2 << "x" << std::endl;
    std::cout << std::endl;

    // Save output images for comparison
    cv::imwrite("blur_naive.png", dst1);
    cv::imwrite("blur_separable.png", dst2);
    std::cout << "Saved output images:" << std::endl;
    std::cout << "  - blur_naive.png" << std::endl;
    std::cout << "  - blur_separable.png" << std::endl;

    // Display images (optional - comment out if running headless)
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Naive Blur", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Separable Blur", cv::WINDOW_AUTOSIZE);

    cv::imshow("Original", src);
    cv::imshow("Naive Blur", dst1);
    cv::imshow("Separable Blur", dst2);

    std::cout << "\nPress any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
