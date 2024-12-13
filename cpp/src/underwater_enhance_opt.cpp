#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Optimized video processing class using Look-Up Tables (LUTs)
 * for faster underwater image enhancement.
 */
class VideoProcessor {
private:
    // Look-up table for float conversion [0-255] -> [0-1]
    cv::Mat float_lut;          
    // Look-up tables for contrast stretching (one per channel)
    cv::Mat stretch_luts[3];    
    double percentile;
    int update_interval;

    /**
     * Initialize Look-Up Tables for float conversion and contrast stretching
     */
    void initializeLUTs() {
        // Initialize float conversion LUT to normalize [0-255] to [0-1]
        float_lut = cv::Mat(1, 256, CV_32FC1);
        float* lutData = float_lut.ptr<float>();
        for (int i = 0; i < 256; i++) {
            lutData[i] = i / 255.0f;
        }
        
        // Initialize LUTs for contrast stretching (one per channel)
        for(int i = 0; i < 3; i++) {
            stretch_luts[i] = cv::Mat(1, 256, CV_8U);
        }
    }

    /**
     * Update contrast stretching LUTs based on current frame histogram
     * @param frame Input BGR frame
     */
    void updateStretchLUTs(const cv::Mat& frame) {
        const int kHistSize = 256;
        int lowerBound = static_cast<int>((100.0 - percentile) * 0.01 * frame.rows * frame.cols);
        
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        
        for(int c = 0; c < 3; ++c) {
            // Calculate histogram for current channel
            int histogram[kHistSize] = {0};
            const uchar* data = channels[c].ptr<uchar>();
            const int totalPixels = channels[c].rows * channels[c].cols;
            
            for(int i = 0; i < totalPixels; ++i) {
                histogram[data[i]]++;
            }
            
            // Find low and high intensity values based on percentile
            int count = 0, low = 0, high = 255;
            for(int i = 0; i < kHistSize; ++i) {
                count += histogram[i];
                if(count >= lowerBound) {
                    low = i;
                    break;
                }
            }
            
            count = 0;
            for(int i = kHistSize - 1; i >= 0; --i) {
                count += histogram[i];
                if(count >= lowerBound) {
                    high = i;
                    break;
                }
            }
            
            // Compute scaling for contrast stretching and update LUT
            uchar* lutData = stretch_luts[c].ptr<uchar>();
            float scale = 255.0f / (high - low);
            for(int i = 0; i < 256; ++i) {
                lutData[i] = cv::saturate_cast<uchar>((i - low) * scale);
            }
        }
    }

public:
    VideoProcessor(double percentile_ = 98.0, int update_interval_ = 30) 
        : percentile(percentile_), update_interval(update_interval_) {
        initializeLUTs();
    }
    
    /**
     * Corrects red channel using optimized LUT-based implementation
     * @param img Input BGR image
     * @return Processed BGR image with corrected red channel
     */
    cv::Mat redCorrection(const cv::Mat& img) {
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        
        // Convert red and green channels to float range [0-1]
        cv::Mat r_float, g_float;
        cv::LUT(channels[2], float_lut, r_float);  // Red channel
        cv::LUT(channels[1], float_lut, g_float);  // Green channel
        
        // Calculate mean difference between green and red channels
        cv::Scalar mean_r = cv::mean(r_float);
        cv::Scalar mean_g = cv::mean(g_float);
        float diff = mean_g[0] - mean_r[0];

        // Adjust red channel proportionally to green channel
        cv::Mat correction = diff * (1.0f - r_float).mul(g_float);
        r_float += correction;
        
        // Convert corrected red channel back to 8-bit [0-255]
        r_float *= 255.0f;
        r_float.convertTo(channels[2], CV_8U);
        
        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }
    
    /**
     * Enhances contrast using pre-computed LUTs
     * @param img Input BGR image
     * @return Processed BGR image with enhanced contrast
     */
    cv::Mat contrastStretch(const cv::Mat& img) {
        static int frame_count = 0;
        // Periodically update LUTs based on update interval
        if (frame_count++ % update_interval == 0) {
            updateStretchLUTs(img);
        }
        
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        
        // Apply per-channel LUTs to stretch contrast
        for(int c = 0; c < 3; ++c) {
            cv::LUT(channels[c], stretch_luts[c], channels[c]);
        }
        
        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }
};

void processImage(const std::string& input, const std::string& output) {
    cv::Mat img = cv::imread(input);
    if (img.empty()) {
        throw std::runtime_error("Could not open input image");
    }

    std::cout << "Processing image..." << std::endl;
    VideoProcessor processor;
    cv::Mat processed = processor.redCorrection(img);
    processed = processor.contrastStretch(processed);
    
    if (!cv::imwrite(output, processed)) {
        throw std::runtime_error("Could not save output image");
    }
    std::cout << "Processed image saved to " << output << std::endl;
}

void processVideo(const std::string& input, const std::string& output) {
    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open input video");
    }

    cv::VideoWriter writer(
        output,
        cv::VideoWriter::fourcc('m','p','4','v'),
        cap.get(cv::CAP_PROP_FPS),
        cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), 
                cap.get(cv::CAP_PROP_FRAME_HEIGHT))
    );

    if (!writer.isOpened()) {
        throw std::runtime_error("Could not create output video");
    }

    VideoProcessor processor;
    cv::Mat frame;
    int frameCount = 0;
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Processing video: " << totalFrames << " frames" << std::endl;

    while (cap.read(frame)) {
        cv::Mat processed = processor.redCorrection(frame);
        processed = processor.contrastStretch(processed);
        writer.write(processed);
        frameCount++;
        if (frameCount % 10 == 0 || frameCount == totalFrames) {
            std::cout << "Processed " << frameCount << "/" << totalFrames << " frames\r";
            std::cout.flush();
        }
    }

    cap.release();
    writer.release();
    std::cout << "\nProcessed video saved to " << output << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_path> <output_path>" << std::endl;
        return -1;
    }

    try {
        std::string input = argv[1];
        std::string ext = input.substr(input.find_last_of('.'));
        
        // Check if input is image or video
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            processImage(argv[1], argv[2]);
        } else if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
            processVideo(argv[1], argv[2]);
        } else {
            throw std::runtime_error("Unsupported file format");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}