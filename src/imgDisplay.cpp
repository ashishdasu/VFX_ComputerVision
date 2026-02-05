// Ashish Dasu
// January 2026
// Read an image from a file, display it in a window, and wait for input.

#include <opencv2/opencv.hpp>
#include <iostream>

/*
 * Reads an image file specified as a command-line argument and displays it.
 * The program enters a loop waiting for keypresses until 'q' is pressed.
 *
 * argv[1] should be the image path
 * returns: 0 on success, -1 on error
 */
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
