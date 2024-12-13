import cv2
import argparse
import os
from image_processor import process_frame

def process_image(input_path, output_path):
    """
    Process a single image file using underwater enhancement algorithms.

    Args:
        input_path (str): Path to input image file
        output_path (str): Path where processed image will be saved
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    img_processed = process_frame(img)
    cv2.imwrite(output_path, img_processed)
    print(f"Processed image saved to {output_path}")

def process_video(input_path, output_path):
    """
    Process a video file frame by frame using underwater enhancement algorithms.

    Args:
        input_path (str): Path to input video file
        output_path (str): Path where processed video will be saved
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {frame_count} frames")

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        img_processed = process_frame(frame)
        out.write(img_processed)
        if (i + 1) % 10 == 0 or (i + 1) == frame_count:
            print(f"Processed {i + 1}/{frame_count} frames", end='\r')

    cap.release()
    out.release()
    print(f"\nProcessed video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Underwater Image Enhancement Tool')
    parser.add_argument('input', help='Path to the input image or video file.')
    parser.add_argument('output', help='Path where the processed file will be saved.')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    # Determine if input is image or video based on file extension
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    ext = os.path.splitext(args.input)[1].lower()
    if ext in image_extensions:
        process_image(args.input, args.output)
    elif ext in video_extensions:
        process_video(args.input, args.output)
    else:
        print("Unsupported file format.")

if __name__ == '__main__':
    main()