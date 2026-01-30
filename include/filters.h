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

/*
 * Applies a depth-based fog effect using depth map information.
 *
 * Creates atmospheric fog that increases with distance from the camera.
 * Uses exponential fog model: fog_amount = 1 - exp(-depth * density)
 *
 * The fog blends the original color with a fog color based on depth:
 *   final_color = original_color * (1 - fog) + fog_color * fog
 *
 * src: input color image (CV_8UC3)
 * depth: depth map (CV_8UC1, greyscale where brighter = farther)
 * dst: output image with fog applied (CV_8UC3)
 * density: fog density parameter (typical: 0.003 - 0.01)
 * fogColor: color of the fog (default: light grey/white)
 * returns: 0 on success
 */
int applyDepthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst,
                   float density = 0.005,
                   cv::Scalar fogColor = cv::Scalar(200, 200, 200));

/*
 * Applies an emboss effect to create a 3D raised surface appearance.
 *
 * Uses a directional gradient kernel that emphasizes edges in one direction
 * while suppressing the opposite direction. This creates the illusion of
 * light coming from one corner, making the image appear raised/stamped.
 *
 * The kernel used is:
 *   [-2 -1  0]
 *   [-1  1  1]
 *   [ 0  1  2]
 *
 * A constant value (128) is added to center the result, since emboss
 * produces both positive and negative values. The result appears as
 * a grayscale relief map with highlights and shadows.
 *
 * src: input color image (CV_8UC3)
 * dst: output embossed image (CV_8UC3)
 * returns: 0 on success
 */
int emboss(cv::Mat &src, cv::Mat &dst);

/*
 * Inverts all color values to create a negative image effect.
 *
 * For each channel: new_value = 255 - old_value
 *
 * This creates an artistic "film negative" effect where:
 *   - Dark areas become bright
 *   - Bright areas become dark
 *   - Colors become complementary (red ↔ cyan, green ↔ magenta, blue ↔ yellow)
 *
 * src: input color image (CV_8UC3)
 * dst: output inverted image (CV_8UC3)
 * returns: 0 on success
 */
int negative(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a vignette effect that darkens the edges and corners of the image.
 *
 * Creates a professional photographic effect where the center remains bright
 * and the image gradually darkens toward the edges. This draws the viewer's
 * attention to the center of the frame.
 *
 * Algorithm:
 *   1. Calculate distance of each pixel from image center
 *   2. Normalize distance to [0, 1] range
 *   3. Apply darkening factor based on distance
 *   4. Blend darkened pixel with original
 *
 * The effect is controlled by:
 *   - strength: how dark the edges become (0.0 - 1.0)
 *   - radius: how far from center the darkening starts (0.0 - 1.0)
 *
 * src: input color image (CV_8UC3)
 * dst: output vignetted image (CV_8UC3)
 * strength: vignette intensity (default: 0.5)
 * radius: inner radius before darkening begins (default: 0.5)
 * returns: 0 on success
 */
int vignette(cv::Mat &src, cv::Mat &dst, float strength = 0.5, float radius = 0.5);

/*
 * Applies portrait mode effect: keeps faces in focus, blurs background.
 *
 * Creates iPhone-style portrait mode by:
 *   1. Detecting faces in the image
 *   2. Blurring the entire frame
 *   3. Creating a mask where face regions = 1, background = 0
 *   4. Blending: sharp faces on blurred background
 *
 * The mask is feathered (gradual transition) to avoid hard edges around faces.
 * Face regions are expanded slightly to include head/shoulders.
 *
 * src: input color image (CV_8UC3)
 * faces: vector of face bounding boxes from face detection
 * dst: output portrait mode image (CV_8UC3)
 * blurAmount: amount of background blur (default: 15, must be odd)
 * featherRadius: smoothness of face edge transition (default: 30 pixels)
 * returns: 0 on success
 */
int portraitMode(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst,
                 int blurAmount = 15, int featherRadius = 30);

/*
 * Detects and highlights motion between consecutive video frames.
 *
 * Compares current frame with previous frame to detect movement:
 *   1. Compute absolute difference between frames
 *   2. Convert to greyscale and threshold
 *   3. Dilate to fill gaps in motion regions
 *   4. Overlay motion mask on original frame (highlight in color)
 *
 * Useful for:
 *   - Security cameras (detect intruders)
 *   - Activity detection (motion sensing)
 *   - Gesture tracking
 *
 * This function maintains internal state (previous frame) using static variables,
 * so it should only be called from one video stream at a time.
 *
 * currentFrame: current video frame (CV_8UC3)
 * dst: output with motion highlighted (CV_8UC3)
 * threshold: sensitivity (0-255, lower = more sensitive, default: 30)
 * highlightColor: color to highlight motion regions (default: red)
 * returns: 0 on success
 */
int motionDetect(cv::Mat &currentFrame, cv::Mat &dst,
                 int threshold = 30,
                 cv::Scalar highlightColor = cv::Scalar(0, 0, 255));

#endif
