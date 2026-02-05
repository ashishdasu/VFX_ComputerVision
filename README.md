# CS5330 Project 1: Video Special Effects

**Author:** Ashish Dasu
**Date:** January 2026

##
Report was too large to submit on gradescope. Here is drive link: https://drive.google.com/file/d/1jJ1MBBd-wb1weq9kyVxRMrzClRPqnn7o/view?usp=share_link


## Video Demonstrations
- **Face Detection:** https://drive.google.com/file/d/1YEfiqKqx2T9_IYnf2Hp97H1clbnEeQTy/view?usp=share_link 
- **Depth Effects (Task 11):** https://drive.google.com/file/d/1jVT1INdXstnupqqymAXq9MvHdvNepiMJ/view?usp=share_link

## Source Files

**src/imgDisplay.cpp** - Reads and displays a static image from file (Task 1)

**src/vidDisplay.cpp** - Main video filter application with real-time effects controlled by keypresses (Tasks 2-10, 12)

**src/depthDemo.cpp** - Standalone depth estimation demo using Depth Anything V2 neural network (Task 11)

**src/filters.cpp** - Implements all filter functions (greyscale, sepia, blur, sobel, face detection, depth effects, etc.)

**src/processSobel.cpp** - Batch utility to apply Sobel edge detection to static images

**src/processBlurQuantize.cpp** - Batch utility to apply blur+quantize cartoon effect to static images

**src/blurTiming.cpp** - Performance benchmark comparing naive vs. separable blur implementations

**include/filters.h** - Header file declaring all filter functions

## Use of LLM Assistance
Used LLM for report template, stitching and labeling cathedral photos, and formatting comments (function descriptions). Also, for help understanding general concepts for the project.


## Additional Note
Got the flu pretty bad for a week. This, unfortunately, meant I could not complete the project within the extension task deadline. Will make sure I have enough time to be more thorough for the next project. Will also use less templating assistance.