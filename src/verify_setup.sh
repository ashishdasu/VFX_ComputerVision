#!/bin/bash
# Setup verification script for OpenCV + ONNX Runtime project
# Run this to check if everything is installed correctly

echo "========================================="
echo "CV Project Setup Verification"
echo "========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ((ERRORS++))
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

echo "1. Checking Homebrew..."
if command -v brew &> /dev/null; then
    print_status 0 "Homebrew is installed"
    BREW_PREFIX=$(brew --prefix)
    echo "   Homebrew prefix: $BREW_PREFIX"
else
    print_status 1 "Homebrew is NOT installed"
    echo "   Install from: https://brew.sh"
fi
echo ""

echo "2. Checking OpenCV..."
if brew list opencv &> /dev/null; then
    print_status 0 "OpenCV is installed via Homebrew"
    OPENCV_VERSION=$(brew list --versions opencv | awk '{print $2}')
    echo "   Version: $OPENCV_VERSION"
    
    # Check for OpenCV libraries
    if [ -f "$BREW_PREFIX/lib/libopencv_core.dylib" ]; then
        print_status 0 "OpenCV core library found"
    else
        print_status 1 "OpenCV core library NOT found"
    fi
    
    # Check for OpenCV headers
    if [ -d "$BREW_PREFIX/include/opencv4" ]; then
        print_status 0 "OpenCV headers found"
    else
        print_status 1 "OpenCV headers NOT found"
    fi
else
    print_status 1 "OpenCV is NOT installed"
    echo "   Install with: brew install opencv"
fi
echo ""

echo "3. Checking ONNX Runtime..."
if brew list onnxruntime &> /dev/null; then
    print_status 0 "ONNX Runtime is installed via Homebrew"
    ONNX_VERSION=$(brew list --versions onnxruntime | awk '{print $2}')
    echo "   Version: $ONNX_VERSION"
    
    # Check for ONNX Runtime library
    if [ -f "$BREW_PREFIX/lib/libonnxruntime.dylib" ]; then
        print_status 0 "ONNX Runtime library found"
    else
        print_status 1 "ONNX Runtime library NOT found"
    fi
    
    # Check for ONNX Runtime headers
    if [ -d "$BREW_PREFIX/Cellar/onnxruntime" ]; then
        print_status 0 "ONNX Runtime headers found"
    else
        print_status 1 "ONNX Runtime headers NOT found"
    fi
else
    print_status 1 "ONNX Runtime is NOT installed"
    echo "   Install with: brew install onnxruntime"
fi
echo ""

echo "4. Checking required OpenCV modules..."
REQUIRED_LIBS=("libopencv_core.dylib" "libopencv_highgui.dylib" "libopencv_video.dylib" 
               "libopencv_videoio.dylib" "libopencv_imgcodecs.dylib" "libopencv_imgproc.dylib" 
               "libopencv_objdetect.dylib")

for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "$BREW_PREFIX/lib/$lib" ]; then
        print_status 0 "$lib"
    else
        print_status 1 "$lib NOT found"
    fi
done
echo ""

echo "5. Checking C++ compiler..."
if command -v clang++ &> /dev/null; then
    print_status 0 "clang++ is available"
    CLANG_VERSION=$(clang++ --version | head -n 1)
    echo "   $CLANG_VERSION"
else
    print_status 1 "clang++ is NOT available"
fi
echo ""

echo "6. Checking project directory structure..."
PROJECT_DIRS=("bin" "include" "src" "data")
for dir in "${PROJECT_DIRS[@]}"; do
    if [ -d "../$dir" ]; then
        print_status 0 "Directory ../$dir exists"
    else
        print_warning "Directory ../$dir does NOT exist (will be created when needed)"
    fi
done
echo ""

echo "7. Checking for makefile..."
if [ -f "makefile" ] || [ -f "Makefile" ]; then
    print_status 0 "makefile found in current directory"
else
    print_warning "makefile NOT found in current directory"
    echo "   Make sure you're running this from the src/ directory"
fi
echo ""

echo "8. Testing a simple compilation..."
cat > test_opencv.cpp << 'EOF'
#include <opencv2/opencv.hpp>
int main() {
    cv::Mat img(100, 100, CV_8UC3);
    return 0;
}
EOF

if clang++ -std=c++11 -I$BREW_PREFIX/include/opencv4 test_opencv.cpp -L$BREW_PREFIX/lib -lopencv_core -o test_opencv 2>/dev/null; then
    print_status 0 "Test compilation successful"
    rm -f test_opencv test_opencv.cpp
else
    print_status 1 "Test compilation FAILED"
    echo "   There may be linking issues"
    rm -f test_opencv.cpp
fi
echo ""

echo "9. Checking for protobuf (common issue)..."
if brew list protobuf &> /dev/null; then
    print_status 0 "protobuf is installed"
    
    # Check if the specific version OpenCV needs exists
    PROTOBUF_LIB=$(find $BREW_PREFIX/lib -name "libprotobuf.*.dylib" 2>/dev/null | head -n 1)
    if [ -n "$PROTOBUF_LIB" ]; then
        print_status 0 "protobuf library found: $(basename $PROTOBUF_LIB)"
    else
        print_warning "protobuf library file not found, may need: brew reinstall protobuf"
    fi
else
    print_status 1 "protobuf is NOT installed"
    echo "   Install with: brew install protobuf"
fi
echo ""

echo "========================================="
echo "Summary"
echo "========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! You're ready to go.${NC}"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Setup is OK with $WARNINGS warnings.${NC}"
    echo "Warnings are usually not critical."
else
    echo -e "${RED}Found $ERRORS errors and $WARNINGS warnings.${NC}"
    echo "Please fix the errors before proceeding."
fi
echo ""

echo "Recommended fixes for common issues:"
echo "  - Missing libraries: brew install opencv onnxruntime protobuf"
echo "  - Protobuf issues: brew reinstall protobuf && brew reinstall opencv"
echo "  - Missing directories: mkdir -p ../bin ../include ../data"
echo ""
