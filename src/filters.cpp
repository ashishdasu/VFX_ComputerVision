// Ashish Dasu
// January 2026
// Implementation of image filter functions for video processing effects.

#include "filters.h"

// Custom greyscale filter using inverse red channel
// Makes red areas dark and blue/green areas bright - looks different from OpenCV's greyscale
int greyscale(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Invert red channel and copy to all three channels
            uchar grey = 255 - srcRow[j][2];
            dstRow[j][0] = dstRow[j][1] = dstRow[j][2] = grey;
        }
    }

    return 0;
}

// Sepia filter - gives old photo look with warm brown tones
// Each output channel is a weighted mix of all input channels
int sepia(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Need to save original values before computing new ones
            uchar oldB = srcRow[j][0];
            uchar oldG = srcRow[j][1];
            uchar oldR = srcRow[j][2];

            // Sepia transformation matrix - saturate_cast prevents overflow
            dstRow[j][0] = cv::saturate_cast<uchar>(0.272*oldR + 0.534*oldG + 0.131*oldB);
            dstRow[j][1] = cv::saturate_cast<uchar>(0.349*oldR + 0.686*oldG + 0.168*oldB);
            dstRow[j][2] = cv::saturate_cast<uchar>(0.393*oldR + 0.769*oldG + 0.189*oldB);
        }
    }

    return 0;
}

// Naive 5x5 blur - applies full 2D kernel with .at() method
// Slower than separable version but more straightforward
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    dst = src.clone();  // Copy for borders

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // Skip border (need 2 pixels in each direction for 5x5 kernel)
    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply full 5x5 kernel
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
                    int weight = kernel[ki + 2][kj + 2];
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Divide by 100 (sum of all kernel weights)
            dst.at<cv::Vec3b>(i, j)[0] = sumB / 100;
            dst.at<cv::Vec3b>(i, j)[1] = sumG / 100;
            dst.at<cv::Vec3b>(i, j)[2] = sumR / 100;
        }
    }

    return 0;
}

// Optimized 5x5 blur using separable filters
// Apply horizontal blur, then vertical - much faster than 2D kernel
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    cv::Mat temp;
    temp.create(src.size(), src.type());
    int kernel1D[5] = {1, 2, 4, 2, 1};

    // Pass 1: blur horizontally
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.cols - 2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -2; k <= 2; k++) {
                sumB += srcRow[j + k][0] * kernel1D[k + 2];
                sumG += srcRow[j + k][1] * kernel1D[k + 2];
                sumR += srcRow[j + k][2] * kernel1D[k + 2];
            }

            tempRow[j][0] = sumB / 10;
            tempRow[j][1] = sumG / 10;
            tempRow[j][2] = sumR / 10;
        }

        // Copy borders
        tempRow[0] = srcRow[0];
        tempRow[1] = srcRow[1];
        tempRow[src.cols - 2] = srcRow[src.cols - 2];
        tempRow[src.cols - 1] = srcRow[src.cols - 1];
    }

    // Pass 2: blur vertically
    dst.create(src.size(), src.type());

    for (int i = 2; i < temp.rows - 2; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < temp.cols; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -2; k <= 2; k++) {
                const cv::Vec3b *tempRowK = temp.ptr<cv::Vec3b>(i + k);
                sumB += tempRowK[j][0] * kernel1D[k + 2];
                sumG += tempRowK[j][1] * kernel1D[k + 2];
                sumR += tempRowK[j][2] * kernel1D[k + 2];
            }

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

// Sobel X - detects vertical edges by measuring horizontal gradient
// Separable implementation: smooth with [1 2 1] vertically, then derivative [-1 0 1] horizontally
// Output is CV_16SC3 (signed short) because gradients can be negative
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // Pass 1: smooth vertically with [1 2 1]
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 0; j < src.cols; j++) {
            const cv::Vec3b *rowAbove = src.ptr<cv::Vec3b>(i - 1);
            const cv::Vec3b *rowCenter = src.ptr<cv::Vec3b>(i);
            const cv::Vec3b *rowBelow = src.ptr<cv::Vec3b>(i + 1);

            // Apply vertical kernel
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

// Sobel Y - detects horizontal edges by measuring vertical gradient
// Separable: smooth with [1 2 1] horizontally, then derivative [-1 0 1] vertically
// Output is CV_16SC3 (signed short) for negative gradients
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // Pass 1: smooth horizontally with [1 2 1]
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

// Gradient magnitude - combines Sobel X and Y to detect all edges regardless of direction
// Uses Euclidean distance: magnitude = sqrt(sx^2 + sy^2)
// Think of (sx,sy) as a vector - magnitude is the length of that vector
// Output is CV_8UC3 (unsigned) since magnitude is always positive
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
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

// Blur and quantize - creates cartoon/poster effect
// First blur to smooth, then reduce colors to discrete levels using integer division
// With levels=10: bucket_size=25, so value 127 -> bucket 5 -> quantized to 125
// Blurring first prevents noisy speckled result
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    cv::Mat blurred;
    blur5x5_2(src, blurred);

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

// Depth-based fog - atmospheric effect using depth map from Depth Anything V2
// Farther objects (brighter in depth map) get more fog applied
// Uses linear fog model: fog_amount = (depth/255) * density, then blend with fog color
// Physically accurate would be exponential (1 - exp(-depth*density)) but linear is faster and looks good
// Typical density: 0.003-0.01 (lower = subtle fog, higher = dense fog)
// Topographic depth contours - like elevation lines on a map
// Draws contour lines at regular depth intervals
int depthContours(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, int numLevels) {
    dst = src.clone();

    // Resize depth if needed
    cv::Mat depthResized;
    if (depth.size() != src.size()) {
        cv::resize(depth, depthResized, src.size());
    } else {
        depthResized = depth;
    }

    // Create contour lines at regular depth intervals
    int levelStep = 255 / numLevels;  // Divide depth range into levels

    for (int level = 1; level < numLevels; level++) {
        int depthThreshold = level * levelStep;

        // Find pixels near this depth threshold
        for (int i = 1; i < depthResized.rows - 1; i++) {
            const uchar *depthRow = depthResized.ptr<uchar>(i);
            cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

            for (int j = 1; j < depthResized.cols - 1; j++) {
                int currentDepth = depthRow[j];

                // Check if any neighbor crosses this depth threshold
                // (creates contour line at depth boundaries)
                bool onContour = false;
                for (int di = -1; di <= 1 && !onContour; di++) {
                    for (int dj = -1; dj <= 1 && !onContour; dj++) {
                        if (di == 0 && dj == 0) continue;

                        int neighborDepth = depthResized.ptr<uchar>(i + di)[j + dj];

                        // If current and neighbor are on opposite sides of threshold
                        if ((currentDepth < depthThreshold && neighborDepth >= depthThreshold) ||
                            (currentDepth >= depthThreshold && neighborDepth < depthThreshold)) {
                            onContour = true;
                        }
                    }
                }

                if (onContour) {
                    // Draw contour line in dark blue
                    dstRow[j] = cv::Vec3b(139, 69, 19);  // Brown contour lines
                }
            }
        }
    }

    return 0;
}

// Color by depth - apply color gradient based on distance
// Near objects = warm colors (red/orange), far objects = cool colors (blue/purple)
int colorByDepth(cv::Mat &src, cv::Mat &depth, cv::Mat &dst) {
    dst.create(src.size(), src.type());

    // Resize depth if needed
    cv::Mat depthResized;
    if (depth.size() != src.size()) {
        cv::resize(depth, depthResized, src.size());
    } else {
        depthResized = depth;
    }

    // Apply color map: use depth value to create color gradient
    cv::Mat depthColor;
    cv::applyColorMap(depthResized, depthColor, cv::COLORMAP_JET);
    // JET colormap: blue (close) -> cyan -> green -> yellow -> red (far)

    // Blend original image with depth color (50/50 mix)
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        const cv::Vec3b *colorRow = depthColor.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Blend original with depth color
            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = (srcRow[j][c] * 0.5 + colorRow[j][c] * 0.5);
            }
        }
    }

    return 0;
}

// Emboss - creates 3D raised surface look using directional gradient kernel
// Positive weights on lower-right, negative on upper-left (simulates light from upper-left)
// Add 128 to center result: flat areas = gray, edges with gradient = bright/dark
int emboss(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());

    int kernel[3][3] = {
        {-2, -1,  0},
        {-1,  1,  1},
        { 0,  1,  2}
    };

    // Process pixels (skip 1-pixel border)
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            // Accumulate weighted pixel values for each channel
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply 3x3 kernel
            for (int ki = -1; ki <= 1; ki++) {
                cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i + ki);
                for (int kj = -1; kj <= 1; kj++) {
                    int weight = kernel[ki + 1][kj + 1];
                    sumB += srcRow[j + kj][0] * weight;
                    sumG += srcRow[j + kj][1] * weight;
                    sumR += srcRow[j + kj][2] * weight;
                }
            }

            // Add 128 to center around middle gray (since emboss produces negative values)
            // This makes flat areas gray, raised edges brighter, recessed edges darker
            dstRow[j][0] = cv::saturate_cast<uchar>(sumB + 128);
            dstRow[j][1] = cv::saturate_cast<uchar>(sumG + 128);
            dstRow[j][2] = cv::saturate_cast<uchar>(sumR + 128);
        }
    }

    // Handle borders (set to middle gray)
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        if (i == 0 || i == src.rows - 1) {
            // Top and bottom rows
            for (int j = 0; j < src.cols; j++) {
                dstRow[j][0] = dstRow[j][1] = dstRow[j][2] = 128;
            }
        } else {
            // Left and right edges
            dstRow[0][0] = dstRow[0][1] = dstRow[0][2] = 128;
            dstRow[src.cols-1][0] = dstRow[src.cols-1][1] = dstRow[src.cols-1][2] = 128;
        }
    }

    return 0;
}

// Negative - inverts all colors (like film negative)
// Simple: new = 255 - old for each channel
// Black becomes white, colors become complementary (red -> cyan, etc.)
int negative(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());

    // Process each pixel
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Invert each channel: new = 255 - old
            dstRow[j][0] = 255 - srcRow[j][0];  // Blue
            dstRow[j][1] = 255 - srcRow[j][1];  // Green
            dstRow[j][2] = 255 - srcRow[j][2];  // Red
        }
    }

    return 0;
}

// Vignette - darkens edges/corners to draw focus to center (common in photography)
// Calculate pixel distance from center, darken based on distance beyond radius threshold
// Use Euclidean distance normalized by max diagonal distance
int vignette(cv::Mat &src, cv::Mat &dst, float strength, float radius) {
    dst.create(src.size(), src.type());

    float centerX = src.cols / 2.0f;
    float centerY = src.rows / 2.0f;
    float maxDist = sqrt(centerX * centerX + centerY * centerY);

    // Process each pixel
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Calculate distance from center
            float dx = j - centerX;
            float dy = i - centerY;
            float dist = sqrt(dx * dx + dy * dy);

            // Normalize distance to [0, 1]
            float distNorm = dist / maxDist;

            // Calculate vignette factor
            // If distance < radius, no darkening (factor = 1.0)
            // If distance > radius, darken based on how far beyond radius
            float vignetteFactor = 1.0f;
            if (distNorm > radius) {
                // How far past the radius (0.0 to 1.0)
                float beyond = (distNorm - radius) / (1.0f - radius);

                // Darkening amount increases with distance
                // factor = 1.0 - (strength * beyond)
                // At the very edge (beyond=1.0), factor = 1.0 - strength
                vignetteFactor = 1.0f - (strength * beyond);
            }

            // Apply vignette by multiplying pixel values by factor
            dstRow[j][0] = cv::saturate_cast<uchar>(srcRow[j][0] * vignetteFactor);
            dstRow[j][1] = cv::saturate_cast<uchar>(srcRow[j][1] * vignetteFactor);
            dstRow[j][2] = cv::saturate_cast<uchar>(srcRow[j][2] * vignetteFactor);
        }
    }

    return 0;
}

// Portrait mode - iPhone-style effect with sharp faces and blurred background
// Blur whole image, create mask for faces, then blend sharp version back on face regions
// Expand face boxes by 30% and feather mask for smooth transitions (no hard edges)
