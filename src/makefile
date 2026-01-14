CC = clang++
CXX = $(CC)

BREW_PREFIX = /opt/homebrew

# Compiler flags
CFLAGS = -std=c++11 -I$(BREW_PREFIX)/include/opencv4 -I../include -I$(BREW_PREFIX)/include/onnxruntime
CXXFLAGS = $(CFLAGS)

# Library paths
LDFLAGS = -L$(BREW_PREFIX)/lib

# OpenCV libraries
OPENCV_LIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio \
              -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

# macOS frameworks
FRAMEWORKS = -framework Cocoa -framework AVFoundation -framework CoreMedia \
             -framework CoreVideo -framework CoreGraphics -framework AppKit

# ONNX Runtime library
ONNX_LIBS = -lonnxruntime

# All libraries combined
LDLIBS = $(OPENCV_LIBS) $(FRAMEWORKS) $(ONNX_LIBS)

# Binary directory
BINDIR = ../bin

# Targets
vid: vidDisplay.o filters.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

face: faceDetect.o filters.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

imod: imgDisplay.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

depth: depthDemo.o filters.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~
