# Video Special Effects

**Author:** Ashish Dasu
**Date:** January 2026

##
Report drive link: https://drive.google.com/file/d/1jJ1MBBd-wb1weq9kyVxRMrzClRPqnn7o/view?usp=share_link

## Video Demonstrations
- **Face Detection:** https://drive.google.com/file/d/1YEfiqKqx2T9_IYnf2Hp97H1clbnEeQTy/view?usp=share_link 
- **Depth Effects:** https://drive.google.com/file/d/1jVT1INdXstnupqqymAXq9MvHdvNepiMJ/view?usp=share_link

## Source Files

**src/imgDisplay.cpp** - Reads and displays a static image from file

**src/vidDisplay.cpp** - Main video filter application with real-time effects controlled by keypresses

**src/depthDemo.cpp** - Standalone depth estimation demo using Depth Anything V2 neural network

**src/filters.cpp** - Implements all filter functions (greyscale, sepia, blur, sobel, face detection, depth effects, etc.)

**src/processSobel.cpp** - Batch utility to apply Sobel edge detection to static images

**src/processBlurQuantize.cpp** - Batch utility to apply blur+quantize cartoon effect to static images

**src/blurTiming.cpp** - Performance benchmark comparing naive vs. separable blur implementations

**include/filters.h** - Header file declaring all filter functions
