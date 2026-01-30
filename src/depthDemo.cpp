/* Ashish Dasu
 * January 2026
 *
 * Demonstration program for Depth Anything V2 network.
 * Captures video and applies depth-based effects.
 *
 * NOTE: Requires instructor-provided DA2Network.hpp and ONNX model file.
 * Replace the placeholder DA2Network.hpp in include/ with the real version.
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

    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Video size: " << refS.width << " x " << refS.height << std::endl;

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Depth Map", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Depth Fog", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    char displayMode = 'd';  // 'd' = depth view, 'f' = fog effect
    int savedCount = 0;

    std::cout << "\nControls:" << std::endl;
    std::cout << "  'd' - Show depth map" << std::endl;
    std::cout << "  'f' - Show fog effect" << std::endl;
    std::cout << "  's' - Save current frame" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;

    for (;;) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        // Process frame with depth network
        // Scale factor 0.5 = half size for faster processing
        depthNet.set_input(frame, 0.5);

        cv::Mat depthMap;
        depthNet.run_network(depthMap, frame.size());

        // Convert depth map to 3-channel for display
        cv::Mat depthDisplay;
        cv::cvtColor(depthMap, depthDisplay, cv::COLOR_GRAY2BGR);

        // Apply depth fog effect
        cv::Mat fogResult;
        applyDepthFog(frame, depthMap, fogResult, 0.005f);

        // Display based on mode
        cv::imshow("Original", frame);
        cv::imshow("Depth Map", depthDisplay);
        cv::imshow("Depth Fog", fogResult);

        // Check for keypress
        char key = cv::waitKey(10);

        if (key == 'q') {
            break;
        } else if (key == 's') {
            // Save all three views
            std::string baseName = "depth_" + std::to_string(savedCount);
            cv::imwrite(baseName + "_original.png", frame);
            cv::imwrite(baseName + "_depth.png", depthDisplay);
            cv::imwrite(baseName + "_fog.png", fogResult);
            std::cout << "Saved: " << baseName << "_*.png" << std::endl;
            savedCount++;
        } else if (key == 'd') {
            displayMode = 'd';
            std::cout << "Mode: Depth Map" << std::endl;
        } else if (key == 'f') {
            displayMode = 'f';
            std::cout << "Mode: Fog Effect" << std::endl;
        }
    }

    return 0;
}
