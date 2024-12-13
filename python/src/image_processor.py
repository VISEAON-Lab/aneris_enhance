import numpy as np
import cv2

# Function to correct red channel based on green channel
# Input: BGR image
# Output: Red-corrected BGR image
def red_correction(img):
    """
    Corrects red channel attenuation in underwater images.
    
    Uses the green channel as reference since it penetrates water better than red.
    Adjusts red values proportionally based on the difference between red and green means.

    Args:
        img (numpy.ndarray): Input BGR image

    Returns:
        numpy.ndarray: Red-corrected BGR image
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    corrected_img = img_rgb.astype(np.float64)/255

    mean_g = np.mean(corrected_img[:,:,1])  # Green channel mean
    mean_r = np.mean(corrected_img[:,:,0])  # Red channel mean

    # Adjust red channel values proportionally
    corrected_img[:,:,0] = corrected_img[:,:,0] + (mean_g - mean_r)*(1-corrected_img[:,:,0])*corrected_img[:,:,1]
    
    # Clip values to 0-1 range and convert back to 8-bit 
    result = (255*corrected_img).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def contrast_stretch(img, prcn=98):
    """
    Enhances image contrast using percentile-based stretching.
    
    Finds the dark and bright thresholds at the given percentile and
    stretches the intensity range to improve visibility.

    Args:
        img (numpy.ndarray): Input BGR image
        prcn (float): Percentile for threshold calculation (default: 98)

    Returns:
        numpy.ndarray: Contrast-enhanced BGR image
    """
    # Calculate high and low percentile thresholds for each channel
    high = np.percentile(img, prcn, axis=(0, 1), keepdims=True)    # High threshold
    low = np.percentile(img, 100 - prcn, axis=(0, 1), keepdims=True)    # Low threshold
    
    # Stretch pixel values between low and high thresholds
    img_stretched = (img - low) / (high - low)
    img_stretched = np.clip(img_stretched, 0, 1) * 255
    return img_stretched.astype(np.uint8)

def process_frame(img, prcn=98):
    """
    Processes a single frame by applying red correction and contrast stretching.

    Args:
        img (numpy.ndarray): Input image frame.
        prcn (float): Percentile for contrast stretching.

    Returns:
        numpy.ndarray: Processed image frame.
    """
    img = red_correction(img)
    img = contrast_stretch(img, prcn)
    return img