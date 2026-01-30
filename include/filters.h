// Ashish Dasu
// January 2026
// Purpose: Header file for image filter functions used in video processing.

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

/*
 * Converts a color image to greyscale using a custom method.
 * This implementation uses the inverse of the red channel copied to all channels,
 * creating a distinct look from the standard luminance-based conversion.
 *
 * src: input color image (BGR format)
 * dst: output greyscale image (BGR format with identical channels)
 * returns: 0 on success
 */
int greyscale(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a sepia tone filter to create an antique photograph effect.
 * Each output channel is computed as a weighted combination of all three input channels
 * using coefficients that shift colors toward warm brown tones.
 *
 * src: input color image (BGR format)
 * dst: output sepia-toned image (BGR format)
 * returns: 0 on success
 */
int sepia(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 5x5 Gaussian blur filter using a naive single-pass approach.
 * Uses the integer approximation kernel:
 *   [1  2  4  2  1]
 *   [2  4  8  4  2]
 *   [4  8 16  8  4]
 *   [2  4  8  4  2]
 *   [1  2  4  2  1]
 *
 * Each pixel is computed by applying the full 2D kernel in one nested loop.
 * Does not modify the outer 2 rows/columns (border handling).
 *
 * src: input color image (BGR format)
 * dst: output blurred image (BGR format)
 * returns: 0 on success
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 5x5 Gaussian blur filter using separable 1x5 filters.
 * Uses two passes with the 1D kernel [1 2 4 2 1]:
 *   Pass 1: Horizontal blur (convolve each row)
 *   Pass 2: Vertical blur (convolve each column)
 *
 * This approach is mathematically equivalent to blur5x5_1 but significantly faster
 * due to reduced number of operations per pixel (10 vs 25 multiplications).
 * Uses row pointers for performance optimization.
 *
 * src: input color image (BGR format)
 * dst: output blurred image (BGR format)
 * returns: 0 on success
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 3x3 Sobel X filter to detect vertical edges (horizontal gradients).
 * Implemented as separable 1x3 filters: [1 2 1] vertical × [-1 0 1] horizontal.
 * Positive response indicates darker-to-brighter transition from left to right.
 *
 * Output is signed 16-bit (CV_16SC3) to capture negative gradient values.
 * Range is [-255*4, 255*4] = [-1020, 1020] due to kernel weights.
 *
 * src: input color image (BGR format, CV_8UC3)
 * dst: output gradient image (signed short, CV_16SC3)
 * returns: 0 on success
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 3x3 Sobel Y filter to detect horizontal edges (vertical gradients).
 * Implemented as separable 1x3 filters: [-1 0 1] vertical × [1 2 1] horizontal.
 * Positive response indicates darker-to-brighter transition from top to bottom.
 *
 * Output is signed 16-bit (CV_16SC3) to capture negative gradient values.
 * Range is [-255*4, 255*4] = [-1020, 1020] due to kernel weights.
 *
 * src: input color image (BGR format, CV_8UC3)
 * dst: output gradient image (signed short, CV_16SC3)
 * returns: 0 on success
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/*
 * Computes gradient magnitude from Sobel X and Y gradient images.
 * Combines directional gradients into a single edge strength map using
 * Euclidean distance: magnitude = sqrt(sx^2 + sy^2)
 *
 * This produces edges regardless of direction - both vertical and horizontal
 * edges appear bright. The magnitude represents the overall rate of change
 * in brightness at each pixel.
 *
 * sx: Sobel X gradient image (CV_16SC3, signed short)
 * sy: Sobel Y gradient image (CV_16SC3, signed short)
 * dst: output magnitude image (CV_8UC3, unsigned char for display)
 * returns: 0 on success
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/*
 * Applies blur followed by color quantization for an artistic effect.
 *
 * Creates a cartoon-like or poster-like appearance by:
 *   1. Blurring the image to reduce noise and detail
 *   2. Quantizing colors to a fixed number of levels
 *
 * Quantization works by dividing the color space into buckets.
 * For example, with levels=10, the range [0,255] is divided into 10 buckets
 * of size 25 each. Each pixel color is rounded to the nearest bucket center.
 *
 * This creates flat color regions with distinct boundaries between them,
 * giving the image a painted or illustrated look.
 *
 * src: input color image (CV_8UC3)
 * dst: output blurred and quantized image (CV_8UC3)
 * levels: number of quantization levels per channel (e.g., 10)
 * returns: 0 on success
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

#endif
