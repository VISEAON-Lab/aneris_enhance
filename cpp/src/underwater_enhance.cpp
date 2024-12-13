#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

/**
 * Corrects red channel attenuation in underwater images.
 * Uses green channel as reference for adjusting red values.
 * @param img Input/output BGR image
 */
void redCorrection(cv::Mat& img) {
    // Corrects the red channel of the image to balance colors.
    // Convert to float and normalize to [0, 1]
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    
    // Split channels (B, G, R)
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    
    // Calculate mean values of red and green channels
    double mean_r = cv::mean(channels[2])[0];   // Red channel mean
    double mean_g = cv::mean(channels[1])[0];   // Green channel mean
    double diff = mean_g - mean_r;

    // Adjust the red channel
    channels[2] += diff * (1.0 - channels[2]).mul(channels[1]);
    
    // Merge channels
    cv::merge(channels, img);
    // Clip values to [0, 1]
    cv::threshold(img, img, 1.0, 1.0, cv::THRESH_TRUNC);
    // Convert back to uint8
    img.convertTo(img, CV_8UC3, 255.0); // Scale back to 0-255 range
}

/**
 * Enhances image contrast using percentile-based stretching.
 * @param img Input/output BGR image
 * @param percentile Threshold percentile for contrast stretch (default: 98.0)
 */
void contrastStretch(cv::Mat& img, double percentile = 98.0) {
    // Performs contrast stretching on the image based on specified percentile.
    // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    for (int i = 0; i < 3; ++i) {
        // Flatten the channel
        cv::Mat flat;
        channels[i].reshape(1, 1).copyTo(flat);

        // Calculate low and high percentile values
        cv::sort(flat, flat, cv::SORT_ASCENDING);
        
        int total_pixels = flat.cols;
        int low_idx = static_cast<int>((100.0 - percentile) / 100.0 * total_pixels);
        int high_idx = static_cast<int>(percentile / 100.0 * total_pixels - 1);
        uchar low_val = flat.at<uchar>(low_idx); // Low threshold
        uchar high_val = flat.at<uchar>(high_idx); // High threshold

        // Stretch pixel values between low and high thresholds
        channels[i].convertTo(channels[i], CV_32F);
        channels[i] = (channels[i] - low_val) / (high_val - low_val) * 255.0;
        channels[i].convertTo(channels[i], CV_8U);
    }

    // Merge channels
    cv::merge(channels, img);
}

void processFrame(cv::Mat& frame) {
    // Convert BGR to RGB for processing
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    redCorrection(frame);
    contrastStretch(frame);
    // Convert back to BGR for output
    cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
}

void processImage(const std::string& inputPath, const std::string& outputPath) {
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "Error: Could not read image " << inputPath << std::endl;
        return;
    }
    processFrame(img);
    cv::imwrite(outputPath, img);
    std::cout << "Processed image saved to " << outputPath << std::endl;
}

void processVideo(const std::string& inputPath, const std::string& outputPath) {
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video " << inputPath << std::endl;
        return;
    }

    int fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open video writer " << outputPath << std::endl;
        return;
    }

    cv::Mat frame;
    int frameCount = 0;
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Processing video: " << totalFrames << " frames" << std::endl;

    while (cap.read(frame)) {
        processFrame(frame);
        writer.write(frame);
        frameCount++;
        if (frameCount % 10 == 0 || frameCount == totalFrames) {
            std::cout << "Processed " << frameCount << "/" << totalFrames << " frames\r";
            std::cout.flush();
        }
    }

    cap.release();
    writer.release();
    std::cout << "\nProcessed video saved to " << outputPath << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./video_processor <input_path> <output_path>" << std::endl;
        return -1;
    }
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    std::string ext = inputPath.substr(inputPath.find_last_of('.'));
    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
        processVideo(inputPath, outputPath);
    } else {
        processImage(inputPath, outputPath);
    }

    return 0;
}