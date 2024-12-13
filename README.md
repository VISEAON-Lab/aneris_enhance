# Underwater Image Enhancement

Implementation of underwater image enhancement techniques in Python and C++.

## Features
- Red channel correction for underwater color compensation
- Contrast stretching for improved visibility
- Support for both image and video processing
- Optimized C++ version using lookup tables (LUTs)

## Requirements

### Python
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

### C++
- OpenCV 4.x
- C++11 compiler

## Usage

### Python

Run the script with the input and output file paths:

```bash
python underwater_enhance.py input_path output_path
```

Examples:

- **Process an image:**

  ```bash
  python underwater_enhance.py input_image.jpg output_image.jpg
  ```

- **Process a video:**

  ```bash
  python underwater_enhance.py input_video.mp4 output_video.mp4
  ```

### C++

Compile the code:

```bash
g++ underwater_enhance.cpp -o underwater_enhance `pkg-config --cflags --libs opencv4`
```

For the optimized version:

```bash
g++ underwater_enhance_opt.cpp -o underwater_enhance_opt `pkg-config --cflags --libs opencv4`
```

Run the executable:

```bash
./underwater_enhance input_path output_path
```

Examples:

- **Process an image:**

  ```bash
  ./underwater_enhance input_image.jpg output_image.jpg
  ```

- **Process a video:**

  ```bash
  ./underwater_enhance input_video.mp4 output_video.mp4
  ```


## Project Structure

```
.
├── python/
│   └── src/
│       ├── underwater_enhance.py
│       └── image_processor.py
└── cpp/
    └── src/
        ├── underwater_enhance.cpp
        └── underwater_enhance_opt.cpp
```

## Performance Metrics

Benchmarks performed on a 2548 × 1440 @ 30 FPS video:

| Implementation | Processing Speed |
|----------------|-----------------|
| Python | 3.7 FPS |
| C++ Standard | 5.0 FPS |
| C++ Optimized (with LUT) | 20 FPS |



### Tested on
- CPU: Intel i7-11800H @ 4.6GHz (8 cores, 16 threads)
- RAM: 32GB DDR4 3200MHz
- OS: Ubuntu 22.04.4 LTS x86_64
- OpenCV: 4.2.0
- Python: 3.10.12
- GCC: 9.4.0
