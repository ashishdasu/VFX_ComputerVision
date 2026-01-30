/* Ashish Dasu
 * January 2026
 *
 * Capture and display live video from webcam with various filter effects.
 * Keypresses activate different filters and functionality.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"

/*
 * Main function
 * Opens the default video device, captures frames in a loop, and displays them.
 * Different keypresses activate various image processing effects.
 *
 * Keypresses:
 *   'q' - quit the program
 *   's' - save current frame to file
 *   'g' - toggle OpenCV greyscale mode
 *   'h' - toggle custom greyscale mode
 *   'e' - toggle sepia tone mode
 *   'b' - toggle blur mode (5x5 Gaussian)
 *   'x' - toggle Sobel X (vertical edges)
 *   'y' - toggle Sobel Y (horizontal edges)
 *   'm' - toggle gradient magnitude (all edges)
 *   'l' - toggle blur+quantize (cartoon effect)
 *
 * argc: number of command-line arguments (unused)
 * argv: array of argument strings (unused)
 * returns: 0 on success, -1 on error
 */
int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // Open the video device (1 = external camera, 0 = built-in)
    capdev = new cv::VideoCapture(1);
    if (!capdev->isOpened()) {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    // Get and display camera properties
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " x " << refS.height << std::endl;

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;
    cv::Mat displayFrame;  // Frame after applying effects

    // Track current display mode
    // 'c' = color (default), 'g' = OpenCV grey, 'h' = custom grey, 'e' = sepia, 'b' = blur
    // 'x' = Sobel X, 'y' = Sobel Y, 'm' = magnitude, 'l' = blur+quantize
    char displayMode = 'c';
    int savedCount = 0;      // Counter for saved images

    // Main capture loop
    for (;;) {
        *capdev >> frame;  // Capture a new frame
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        // Apply the current display mode effect
        switch (displayMode) {
            case 'g':  // OpenCV greyscale
                cv::cvtColor(frame, displayFrame, cv::COLOR_BGR2GRAY);
                // Convert back to 3-channel for consistent display
                cv::cvtColor(displayFrame, displayFrame, cv::COLOR_GRAY2BGR);
                break;
            case 'h':  // Custom greyscale
                greyscale(frame, displayFrame);
                break;
            case 'e':  // Sepia tone
                sepia(frame, displayFrame);
                break;
            case 'b':  // Blur (using optimized separable filter)
                blur5x5_2(frame, displayFrame);
                break;
            case 'x': {  // Sobel X (vertical edges)
                cv::Mat sobelResult;
                sobelX3x3(frame, sobelResult);
                // Convert from signed short to unsigned char for display
                cv::convertScaleAbs(sobelResult, displayFrame);
                break;
            }
            case 'y': {  // Sobel Y (horizontal edges)
                cv::Mat sobelResult;
                sobelY3x3(frame, sobelResult);
                // Convert from signed short to unsigned char for display
                cv::convertScaleAbs(sobelResult, displayFrame);
                break;
            }
            case 'm': {  // Gradient magnitude (all edges)
                cv::Mat sobelX, sobelY;
                sobelX3x3(frame, sobelX);  // Compute X gradient
                sobelY3x3(frame, sobelY);  // Compute Y gradient
                magnitude(sobelX, sobelY, displayFrame);  // Combine into magnitude
                break;
            }
            case 'l':  // Blur and quantize (cartoon effect)
                blurQuantize(frame, displayFrame, 10);  // 10 levels per channel
                break;
            default:   // Color (no effect)
                displayFrame = frame.clone();
                break;
        }

        cv::imshow("Video", displayFrame);

        // Check for keypress (wait 10ms)
        char key = cv::waitKey(10);

        if (key == 'q') {
            // Quit
            break;
        } else if (key == 's') {
            // Save current frame
            std::string filename = "saved_frame_" + std::to_string(savedCount++) + ".png";
            cv::imwrite(filename, displayFrame);
            std::cout << "Saved: " << filename << std::endl;
        } else if (key == 'g') {
            // Toggle OpenCV greyscale
            displayMode = (displayMode == 'g') ? 'c' : 'g';
            std::cout << "Mode: " << (displayMode == 'g' ? "OpenCV Greyscale" : "Color") << std::endl;
        } else if (key == 'h') {
            // Toggle custom greyscale
            displayMode = (displayMode == 'h') ? 'c' : 'h';
            std::cout << "Mode: " << (displayMode == 'h' ? "Custom Greyscale" : "Color") << std::endl;
        } else if (key == 'e') {
            // Toggle sepia tone
            displayMode = (displayMode == 'e') ? 'c' : 'e';
            std::cout << "Mode: " << (displayMode == 'e' ? "Sepia Tone" : "Color") << std::endl;
        } else if (key == 'b') {
            // Toggle blur
            displayMode = (displayMode == 'b') ? 'c' : 'b';
            std::cout << "Mode: " << (displayMode == 'b' ? "Blur" : "Color") << std::endl;
        } else if (key == 'x') {
            // Toggle Sobel X
            displayMode = (displayMode == 'x') ? 'c' : 'x';
            std::cout << "Mode: " << (displayMode == 'x' ? "Sobel X (vertical edges)" : "Color") << std::endl;
        } else if (key == 'y') {
            // Toggle Sobel Y
            displayMode = (displayMode == 'y') ? 'c' : 'y';
            std::cout << "Mode: " << (displayMode == 'y' ? "Sobel Y (horizontal edges)" : "Color") << std::endl;
        } else if (key == 'm') {
            // Toggle gradient magnitude
            displayMode = (displayMode == 'm') ? 'c' : 'm';
            std::cout << "Mode: " << (displayMode == 'm' ? "Gradient Magnitude (all edges)" : "Color") << std::endl;
        } else if (key == 'l') {
            // Toggle blur and quantize
            displayMode = (displayMode == 'l') ? 'c' : 'l';
            std::cout << "Mode: " << (displayMode == 'l' ? "Blur + Quantize (cartoon)" : "Color") << std::endl;
        }
    }

    delete capdev;
    return 0;
}
