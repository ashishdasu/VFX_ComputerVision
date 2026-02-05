/* Ashish Dasu
 * January 2026
 *
 * Captures video and applies depth-based effects.
 *
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include "DA2Network.hpp"
#include "filters.h"

int main(int argc, char *argv[]) {
    // Path to ONNX model
    const char *modelPath = "../data/model_fp16.onnx";

    std::cout << "Initializing Depth Anything V2 network..." << std::endl;
    std::cout << "Model path: " << modelPath << std::endl;

    // Initialize the depth network
    DA2Network depthNet(modelPath);

    // Open video device
    cv::VideoCapture capdev(1);  // 1 = external camera, 0 = built-in
    if (!capdev.isOpened()) {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    // Request 800x600 resolution for balance of quality and performance
    capdev.set(cv::CAP_PROP_FRAME_WIDTH, 800);
    capdev.set(cv::CAP_PROP_FRAME_HEIGHT, 600);

    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Video size: " << refS.width << " x " << refS.height << std::endl;

    // Warm up the camera - discard first few frames
    cv::Mat warmup;
    for (int i = 0; i < 10; i++) {
        capdev >> warmup;
    }

    cv::namedWindow("Depth Demo", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    cv::Mat depthMap;  // Reuse depth map across frames
    char displayMode = 'o';  // 'o' = original, 'd' = depth, 'c' = contours, 'r' = rainbow color
    int savedCount = 0;
    int frameCount = 0;  // For frame skipping

    std::cout << "\nControls:" << std::endl;
    std::cout << "  'o' - Show original video" << std::endl;
    std::cout << "  'd' - Show depth map" << std::endl;
    std::cout << "  'c' - Show depth contours" << std::endl;
    std::cout << "  'r' - Show rainbow depth coloring" << std::endl;
    std::cout << "  's' - Save current frame" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;

    for (;;) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        // Process depth network only every 2 frames for performance
        frameCount++;
        if (frameCount % 2 == 0) {
            // Scale factor 0.25 = 1/4 size for faster processing
            depthNet.set_input(frame, 0.25);
            depthNet.run_network(depthMap, frame.size());
        }

        // Display based on current mode (only compute what we need)
        cv::Mat displayFrame;
        if (depthMap.empty()) {
            // First frame, depth not ready yet
            displayFrame = frame;
        } else if (displayMode == 'd') {
            cv::cvtColor(depthMap, displayFrame, cv::COLOR_GRAY2BGR);
        } else if (displayMode == 'c') {
            depthContours(frame, depthMap, displayFrame, 10);
        } else if (displayMode == 'r') {
            colorByDepth(frame, depthMap, displayFrame);
        } else {
            displayFrame = frame;
        }
        cv::imshow("Depth Demo", displayFrame);

        // Check for keypress
        char key = cv::waitKey(10);

        if (key == 'q') {
            break;
        } else if (key == 's') {
            // Save current display only
            std::string filename = "../report/depth_" + std::to_string(savedCount) + ".png";
            cv::imwrite(filename, displayFrame);
            std::cout << "Saved: " << filename << std::endl;
            savedCount++;
        } else if (key == 'o') {
            displayMode = 'o';
            std::cout << "Mode: Original" << std::endl;
        } else if (key == 'd') {
            displayMode = 'd';
            std::cout << "Mode: Depth Map" << std::endl;
        } else if (key == 'c') {
            displayMode = 'c';
            std::cout << "Mode: Depth Contours" << std::endl;
        } else if (key == 'r') {
            displayMode = 'r';
            std::cout << "Mode: Rainbow Color by Depth" << std::endl;
        }
    }

    return 0;
}
