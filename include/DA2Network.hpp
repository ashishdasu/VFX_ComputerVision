/*
  Wrapper class for Depth Anything V2 network

  This is a PLACEHOLDER - Replace with instructor-provided DA2Network.hpp

  The actual implementation uses ONNX Runtime to run the depth estimation network.
  Expected interface based on assignment description:

  class DA2Network {
  public:
      DA2Network(const std::string& model_path);
      cv::Mat process(const cv::Mat& input);  // Returns normalized depth map [0-255]
  };
*/

#ifndef DA2NETWORK_HPP
#define DA2NETWORK_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

// PLACEHOLDER CLASS
// Replace this entire file with instructor-provided DA2Network.hpp
class DA2Network {
private:
    bool initialized;

public:
    DA2Network(const std::string& model_path) : initialized(false) {
        // TODO: Replace with actual ONNX Runtime initialization
        std::cout << "WARNING: Using placeholder DA2Network" << std::endl;
        std::cout << "Replace with instructor-provided DA2Network.hpp" << std::endl;
    }

    cv::Mat process(const cv::Mat& input) {
        // PLACEHOLDER: Returns a simple gradient as fake depth
        // Real implementation runs the ONNX model
        cv::Mat fake_depth(input.size(), CV_8UC1);

        for (int i = 0; i < input.rows; i++) {
            uchar* row = fake_depth.ptr<uchar>(i);
            for (int j = 0; j < input.cols; j++) {
                // Simple gradient: farther from top = "farther" in depth
                row[j] = (uchar)((i * 255) / input.rows);
            }
        }

        return fake_depth;
    }
};

#endif
