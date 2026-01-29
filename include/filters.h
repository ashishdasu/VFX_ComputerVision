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

#endif
