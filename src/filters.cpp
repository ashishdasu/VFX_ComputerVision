// Ashish Dasu
// January 2026
// Purpose: Implementation of image filter functions for video processing effects.

#include "filters.h"

/*
 * Converts a color image to greyscale using a custom method.
 *
 * Method: Uses (255 - red channel) as the greyscale value.
 * This creates an inverted luminance effect where bright red areas become dark
 * and dark/blue areas become bright, producing a distinctly different result
 * from OpenCV's standard greyscale conversion.
 *
 * OpenCV's cvtColor uses: Y = 0.299*R + 0.587*G + 0.114*B (BT.601 standard)
 * This custom method uses: Y = 255 - R (inverse red channel)
 *
 * src: input color image (BGR format)
 * dst: output image (BGR format with identical R, G, B channels)
 * returns: 0 on success
 */
int greyscale(cv::Mat &src, cv::Mat &dst) {
    // Create output image with same size and type as input
    dst.create(src.size(), src.type());

    // Process each pixel
    for (int i = 0; i < src.rows; i++) {
        // Get pointers to the row data for faster access
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // BGR format: index 0=Blue, 1=Green, 2=Red
            // Use inverse of red channel for unique greyscale effect
            uchar grey = 255 - srcRow[j][2];

            // Set all three channels to the same value
            dstRow[j][0] = grey;  // Blue
            dstRow[j][1] = grey;  // Green
            dstRow[j][2] = grey;  // Red
        }
    }

    return 0;
}

/*
 * Applies a sepia tone filter to a color image.
 *
 * Sepia tone creates an antique photograph effect by shifting all colors toward
 * warm brown/orange tones. Each output channel is a weighted combination of all
 * three input channels using the following coefficients:
 *
 *   Blue  = 0.272*R + 0.534*G + 0.131*B
 *   Green = 0.349*R + 0.686*G + 0.168*B
 *   Red   = 0.393*R + 0.769*G + 0.189*B
 *
 * Important: The red and green coefficients sum to more than 1.0, which means
 * bright pixels can produce values > 255. We use saturate_cast to clamp values
 * to the valid range [0, 255].
 *
 * Also critical: We must use the ORIGINAL R, G, B values for all three calculations.
 * If we overwrite the red channel first, then use the new red value to compute green,
 * the result will be incorrect.
 *
 * src: input color image (BGR format)
 * dst: output sepia-toned image (BGR format)
 * returns: 0 on success
 */
int sepia(cv::Mat &src, cv::Mat &dst) {
    // Create output image with same size and type as input
    dst.create(src.size(), src.type());

    // Process each pixel
    for (int i = 0; i < src.rows; i++) {
        // Get pointers to the row data for faster access
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // BGR format: index 0=Blue, 1=Green, 2=Red
            // Store original values before overwriting
            uchar oldB = srcRow[j][0];
            uchar oldG = srcRow[j][1];
            uchar oldR = srcRow[j][2];

            // Apply sepia transformation using original values
            // saturate_cast clamps the result to [0, 255] to handle overflow
            dstRow[j][0] = cv::saturate_cast<uchar>(0.272*oldR + 0.534*oldG + 0.131*oldB);  // Blue
            dstRow[j][1] = cv::saturate_cast<uchar>(0.349*oldR + 0.686*oldG + 0.168*oldB);  // Green
            dstRow[j][2] = cv::saturate_cast<uchar>(0.393*oldR + 0.769*oldG + 0.189*oldB);  // Red
        }
    }

    return 0;
}
