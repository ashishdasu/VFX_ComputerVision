// Ashish Dasu
// Read and display an image from file

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }
    
    cv::Mat img = cv::imread(argv[1]);
    
    if(img.empty()) {
        std::cout << "Could not read image" << std::endl;
        return -1;
    }
    
    cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display", img);
    
    while(true) {
        char key = cv::waitKey(0);
        if(key == 'q') break;
    }
    
    return 0;
}
