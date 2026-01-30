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

/*
 * Applies a 5x5 Gaussian blur using a naive single-pass approach.
 *
 * This is the straightforward implementation: for each pixel, we apply the full
 * 5x5 kernel by summing weighted values of all 25 neighboring pixels. The kernel
 * is an integer approximation of a Gaussian distribution:
 *
 *   [1  2  4  2  1]
 *   [2  4  8  4  2]
 *   [4  8 16  8  4]    (sum = 100)
 *   [2  4  8  4  2]
 *   [1  2  4  2  1]
 *
 * Each output pixel = (sum of weighted neighbors) / 100
 *
 * Performance: For each pixel, we do 25 array accesses, 25 multiplications,
 * and 25 additions. This is the baseline implementation for comparison.
 *
 * We use the .at() method as requested by the assignment, which includes
 * bounds checking on each access (slower but safer for learning).
 *
 * Border handling: We don't process the outer 2 rows/columns. Instead, we copy
 * the source image to the destination first, then overwrite the interior.
 *
 * src: input color image (BGR format)
 * dst: output blurred image (BGR format)
 * returns: 0 on success
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    // Copy source to destination (handles borders)
    dst = src.clone();

    // Define the 5x5 Gaussian kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // Process interior pixels only (skip 2-pixel border on each side)
    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            // Accumulate weighted sum for each color channel
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply 5x5 kernel centered at (i, j)
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    // Get pixel at offset (ki, kj) from center
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
                    int weight = kernel[ki + 2][kj + 2];  // Kernel indices are 0-4

                    // Accumulate weighted values for each channel
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Divide by sum of kernel weights (100) and write result
            dst.at<cv::Vec3b>(i, j)[0] = sumB / 100;
            dst.at<cv::Vec3b>(i, j)[1] = sumG / 100;
            dst.at<cv::Vec3b>(i, j)[2] = sumR / 100;
        }
    }

    return 0;
}

/*
 * Applies a 5x5 Gaussian blur using separable 1x5 filters.
 *
 * Key insight: A 2D Gaussian kernel can be decomposed into two 1D kernels.
 * Our 5x5 kernel is the outer product of the 1D kernel [1 2 4 2 1] with itself:
 *
 *   [1]
 *   [2]              [1  2  4  2  1]
 *   [4]  *  [1 2 4 2 1]  =  [full 5x5 kernel]
 *   [2]
 *   [1]
 *
 * This means we can:
 *   1. Apply horizontal 1x5 blur: convolve each row with [1 2 4 2 1]
 *   2. Apply vertical 1x5 blur: convolve each column with [1 2 4 2 1]
 *
 * Performance gain:
 *   - Naive: 25 multiplications per pixel
 *   - Separable: 5 + 5 = 10 multiplications per pixel
 *   - Speedup: 2.5x reduction in multiplications
 *
 * We also use row pointers instead of .at() for additional performance gain.
 *
 * src: input color image (BGR format)
 * dst: output blurred image (BGR format)
 * returns: 0 on success
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    // Intermediate image to store horizontal blur result
    cv::Mat temp;
    temp.create(src.size(), src.type());

    // 1D kernel [1 2 4 2 1], sum = 10
    int kernel1D[5] = {1, 2, 4, 2, 1};

    // PASS 1: Horizontal blur (blur each row)
    // For each pixel, convolve with [1 2 4 2 1] horizontally
    for (int i = 0; i < src.rows; i++) {
        // Get row pointers for fast access
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.cols - 2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply 1D horizontal kernel
            for (int k = -2; k <= 2; k++) {
                cv::Vec3b pixel = srcRow[j + k];
                int weight = kernel1D[k + 2];

                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            // Divide by 10 (sum of 1D kernel)
            tempRow[j][0] = sumB / 10;
            tempRow[j][1] = sumG / 10;
            tempRow[j][2] = sumR / 10;
        }

        // Copy border pixels (first 2 and last 2 columns)
        if (i < src.rows) {
            tempRow[0] = srcRow[0];
            tempRow[1] = srcRow[1];
            tempRow[src.cols - 2] = srcRow[src.cols - 2];
            tempRow[src.cols - 1] = srcRow[src.cols - 1];
        }
    }

    // PASS 2: Vertical blur (blur each column)
    // For each pixel, convolve with [1 2 4 2 1] vertically
    dst.create(src.size(), src.type());

    for (int i = 2; i < temp.rows - 2; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < temp.cols; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply 1D vertical kernel
            for (int k = -2; k <= 2; k++) {
                const cv::Vec3b *tempRowK = temp.ptr<cv::Vec3b>(i + k);
                cv::Vec3b pixel = tempRowK[j];
                int weight = kernel1D[k + 2];

                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            // Divide by 10 (sum of 1D kernel)
            dstRow[j][0] = sumB / 10;
            dstRow[j][1] = sumG / 10;
            dstRow[j][2] = sumR / 10;
        }
    }

    // Copy border rows (first 2 and last 2 rows)
    for (int j = 0; j < src.cols; j++) {
        dst.ptr<cv::Vec3b>(0)[j] = temp.ptr<cv::Vec3b>(0)[j];
        dst.ptr<cv::Vec3b>(1)[j] = temp.ptr<cv::Vec3b>(1)[j];
        dst.ptr<cv::Vec3b>(src.rows - 2)[j] = temp.ptr<cv::Vec3b>(src.rows - 2)[j];
        dst.ptr<cv::Vec3b>(src.rows - 1)[j] = temp.ptr<cv::Vec3b>(src.rows - 1)[j];
    }

    return 0;
}

/*
 * Applies a 3x3 Sobel X filter to detect vertical edges.
 *
 * The Sobel X operator detects edges where brightness changes horizontally
 * (left to right). It responds strongly to vertical edges (like vertical lines).
 *
 * Full 3x3 Sobel X kernel:
 *   [-1  0  1]
 *   [-2  0  2]
 *   [-1  0  1]
 *
 * We implement this as separable 1x3 filters:
 *   Vertical smoothing: [1 2 1]^T
 *   Horizontal derivative: [-1 0 1]
 *
 * Two-pass approach:
 *   Pass 1: Apply [1 2 1] vertically (smooths in vertical direction)
 *   Pass 2: Apply [-1 0 1] horizontally (takes horizontal derivative)
 *
 * Output type: CV_16SC3 (signed short) because gradients can be negative.
 * For a white-to-black edge: gradient is negative.
 * For a black-to-white edge: gradient is positive.
 *
 * src: input color image (CV_8UC3)
 * dst: output gradient image (CV_16SC3)
 * returns: 0 on success
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    // Temporary image for vertical smoothing pass (still signed short)
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // PASS 1: Vertical smoothing with [1 2 1]
    // Apply to each column
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 0; j < src.cols; j++) {
            // Get three vertical neighbors
            const cv::Vec3b *rowAbove = src.ptr<cv::Vec3b>(i - 1);
            const cv::Vec3b *rowCenter = src.ptr<cv::Vec3b>(i);
            const cv::Vec3b *rowBelow = src.ptr<cv::Vec3b>(i + 1);

            // Apply [1 2 1] vertically for each channel
            tempRow[j][0] = rowAbove[j][0] * 1 + rowCenter[j][0] * 2 + rowBelow[j][0] * 1;  // Blue
            tempRow[j][1] = rowAbove[j][1] * 1 + rowCenter[j][1] * 2 + rowBelow[j][1] * 1;  // Green
            tempRow[j][2] = rowAbove[j][2] * 1 + rowCenter[j][2] * 2 + rowBelow[j][2] * 1;  // Red
        }
    }

    // Copy border rows (simple approach: set to zero)
    cv::Vec3s *tempRowTop = temp.ptr<cv::Vec3s>(0);
    cv::Vec3s *tempRowBottom = temp.ptr<cv::Vec3s>(src.rows - 1);
    for (int j = 0; j < src.cols; j++) {
        tempRowTop[j] = cv::Vec3s(0, 0, 0);
        tempRowBottom[j] = cv::Vec3s(0, 0, 0);
    }

    // PASS 2: Horizontal derivative with [-1 0 1]
    // Apply to each row
    dst.create(src.size(), CV_16SC3);

    for (int i = 0; i < temp.rows; i++) {
        const cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < temp.cols - 1; j++) {
            // Apply [-1 0 1] horizontally for each channel
            // Note: We don't divide here - keep full gradient magnitude
            dstRow[j][0] = tempRow[j - 1][0] * -1 + tempRow[j][0] * 0 + tempRow[j + 1][0] * 1;  // Blue
            dstRow[j][1] = tempRow[j - 1][1] * -1 + tempRow[j][1] * 0 + tempRow[j + 1][1] * 1;  // Green
            dstRow[j][2] = tempRow[j - 1][2] * -1 + tempRow[j][2] * 0 + tempRow[j + 1][2] * 1;  // Red
        }

        // Border columns set to zero
        dstRow[0] = cv::Vec3s(0, 0, 0);
        dstRow[src.cols - 1] = cv::Vec3s(0, 0, 0);
    }

    return 0;
}

/*
 * Applies a 3x3 Sobel Y filter to detect horizontal edges.
 *
 * The Sobel Y operator detects edges where brightness changes vertically
 * (top to bottom). It responds strongly to horizontal edges (like horizontal lines).
 *
 * Full 3x3 Sobel Y kernel:
 *   [-1 -2 -1]
 *   [ 0  0  0]
 *   [ 1  2  1]
 *
 * We implement this as separable 1x3 filters:
 *   Vertical derivative: [-1 0 1]^T
 *   Horizontal smoothing: [1 2 1]
 *
 * Two-pass approach:
 *   Pass 1: Apply [1 2 1] horizontally (smooths in horizontal direction)
 *   Pass 2: Apply [-1 0 1] vertically (takes vertical derivative)
 *
 * Output type: CV_16SC3 (signed short) to handle negative gradients.
 *
 * src: input color image (CV_8UC3)
 * dst: output gradient image (CV_16SC3)
 * returns: 0 on success
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    // Temporary image for horizontal smoothing pass
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // PASS 1: Horizontal smoothing with [1 2 1]
    // Apply to each row
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            // Apply [1 2 1] horizontally for each channel
            tempRow[j][0] = srcRow[j - 1][0] * 1 + srcRow[j][0] * 2 + srcRow[j + 1][0] * 1;  // Blue
            tempRow[j][1] = srcRow[j - 1][1] * 1 + srcRow[j][1] * 2 + srcRow[j + 1][1] * 1;  // Green
            tempRow[j][2] = srcRow[j - 1][2] * 1 + srcRow[j][2] * 2 + srcRow[j + 1][2] * 1;  // Red
        }

        // Border columns set to zero
        tempRow[0] = cv::Vec3s(0, 0, 0);
        tempRow[src.cols - 1] = cv::Vec3s(0, 0, 0);
    }

    // PASS 2: Vertical derivative with [-1 0 1]
    // Apply to each column
    dst.create(src.size(), CV_16SC3);

    for (int i = 1; i < temp.rows - 1; i++) {
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < temp.cols; j++) {
            // Get three vertical neighbors from temp
            const cv::Vec3s *rowAbove = temp.ptr<cv::Vec3s>(i - 1);
            const cv::Vec3s *rowCenter = temp.ptr<cv::Vec3s>(i);
            const cv::Vec3s *rowBelow = temp.ptr<cv::Vec3s>(i + 1);

            // Apply [-1 0 1] vertically for each channel
            dstRow[j][0] = rowAbove[j][0] * -1 + rowCenter[j][0] * 0 + rowBelow[j][0] * 1;  // Blue
            dstRow[j][1] = rowAbove[j][1] * -1 + rowCenter[j][1] * 0 + rowBelow[j][1] * 1;  // Green
            dstRow[j][2] = rowAbove[j][2] * -1 + rowCenter[j][2] * 0 + rowBelow[j][2] * 1;  // Red
        }
    }

    // Border rows set to zero
    cv::Vec3s *dstRowTop = dst.ptr<cv::Vec3s>(0);
    cv::Vec3s *dstRowBottom = dst.ptr<cv::Vec3s>(dst.rows - 1);
    for (int j = 0; j < src.cols; j++) {
        dstRowTop[j] = cv::Vec3s(0, 0, 0);
        dstRowBottom[j] = cv::Vec3s(0, 0, 0);
    }

    return 0;
}

/*
 * Computes gradient magnitude from Sobel X and Y images.
 *
 * The gradient magnitude combines the X and Y directional gradients into
 * a single measure of edge strength. It answers: "how much is brightness
 * changing at this pixel, regardless of direction?"
 *
 * Formula: magnitude = sqrt(sx^2 + sy^2)
 *
 * This is the Euclidean distance (L2 norm) of the gradient vector.
 * Geometrically, if you think of (sx, sy) as a 2D vector pointing in the
 * direction of maximum brightness change, the magnitude is the length of
 * that vector.
 *
 * Why this works:
 * - Sobel X alone only detects vertical edges
 * - Sobel Y alone only detects horizontal edges
 * - Magnitude detects ALL edges (combines both directions)
 *
 * Example:
 *   Vertical edge: sx=400, sy=0   → magnitude = 400
 *   Horizontal edge: sx=0, sy=400 → magnitude = 400
 *   Diagonal edge: sx=300, sy=300 → magnitude = 424 (sqrt(300^2 + 300^2))
 *
 * We compute this per color channel, so edges in any color are detected.
 *
 * sx: Sobel X gradient image (CV_16SC3, signed short)
 * sy: Sobel Y gradient image (CV_16SC3, signed short)
 * dst: output magnitude image (CV_8UC3, unsigned char)
 * returns: 0 on success
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // Create output image (unsigned char for display)
    dst.create(sx.size(), CV_8UC3);

    // Process each pixel
    for (int i = 0; i < sx.rows; i++) {
        // Get row pointers (signed short for input, unsigned char for output)
        const cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        const cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            // Compute magnitude for each color channel
            // magnitude = sqrt(sx^2 + sy^2)

            // Blue channel
            int sxValB = sxRow[j][0];  // X gradient (signed)
            int syValB = syRow[j][0];  // Y gradient (signed)
            int magB = sqrt(sxValB * sxValB + syValB * syValB);

            // Green channel
            int sxValG = sxRow[j][1];
            int syValG = syRow[j][1];
            int magG = sqrt(sxValG * sxValG + syValG * syValG);

            // Red channel
            int sxValR = sxRow[j][2];
            int syValR = syRow[j][2];
            int magR = sqrt(sxValR * sxValR + syValR * syValR);

            // Clamp to [0, 255] and store in output
            // saturate_cast handles clamping automatically
            dstRow[j][0] = cv::saturate_cast<uchar>(magB);
            dstRow[j][1] = cv::saturate_cast<uchar>(magG);
            dstRow[j][2] = cv::saturate_cast<uchar>(magR);
        }
    }

    return 0;
}

/*
 * Applies blur and color quantization for an artistic effect.
 *
 * This function creates a cartoon-like or poster-art appearance by:
 *   1. Blurring to smooth out noise and fine details
 *   2. Reducing the number of distinct colors to a fixed set
 *
 * Color quantization works by dividing the [0, 255] range into discrete levels.
 * For example, with levels=10:
 *   - Bucket size = 255/10 = 25 (integer division)
 *   - Original values map to: 0, 25, 50, 75, 100, 125, 150, 175, 200, 225
 *
 * Algorithm for quantization:
 *   1. bucket_size = 255 / levels
 *   2. bucket_index = pixel_value / bucket_size   (integer division)
 *   3. quantized_value = bucket_index * bucket_size
 *
 * Example with levels=10:
 *   - Value 127: bucket_index = 127/25 = 5, quantized = 5*25 = 125
 *   - Value 60:  bucket_index = 60/25 = 2,  quantized = 2*25 = 50
 *   - Value 240: bucket_index = 240/25 = 9, quantized = 9*25 = 225
 *
 * The blur step first smooths the image so that quantization creates
 * larger uniform color regions rather than noisy speckled areas.
 *
 * src: input color image (CV_8UC3)
 * dst: output blurred and quantized image (CV_8UC3)
 * levels: number of quantization levels per channel (typical: 8-15)
 * returns: 0 on success
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    // Step 1: Blur the image to reduce noise and smooth details
    cv::Mat blurred;
    blur5x5_2(src, blurred);  // Use optimized separable blur

    // Step 2: Quantize the blurred image
    dst.create(src.size(), src.type());

    // Calculate bucket size for quantization
    int bucketSize = 255 / levels;

    // Process each pixel
    for (int i = 0; i < blurred.rows; i++) {
        const cv::Vec3b *blurRow = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < blurred.cols; j++) {
            // Quantize each channel independently
            for (int c = 0; c < 3; c++) {
                uchar originalValue = blurRow[j][c];

                // Find which bucket this value falls into
                int bucketIndex = originalValue / bucketSize;

                // Map to the bucket's representative value
                uchar quantizedValue = bucketIndex * bucketSize;

                dstRow[j][c] = quantizedValue;
            }
        }
    }

    return 0;
}
