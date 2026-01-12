import os
import cv2
import numpy as np
from PIL import Image
import pytesseract


# Optional for Windows users
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_word_images(image_path, min_confidence=60):
    """
    Extracts individual word images from an input image using OCR.
    Returns a list of PIL Image objects.
    """
    image = Image.open(image_path)

    # Get detailed OCR data including bounding boxes
    ocr_data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    word_images = []

    num_items = len(ocr_data["text"])

    for i in range(num_items):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])

        # Skip empty or low-confidence detections
        if text == "" or conf < min_confidence:
            continue

        x = ocr_data["left"][i]
        y = ocr_data["top"][i]
        w = ocr_data["width"][i]
        h = ocr_data["height"][i]

        # Crop the word region
        word_image = image.crop((x, y, x + w, y + h))
        word_images.append(word_image)

    return word_images

def extract_characters(word_image_pil, output_dir):
    import cv2
    import numpy as np
    import os

    # Convert PIL image to OpenCV grayscale
    img = np.array(word_image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalize contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Global black-white threshold (Otsu)
    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological closing to merge broken character parts
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Light dilation to prevent character splits
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.dilate(bw, dilate_kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        bw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours
    boxes = []
    img_h, img_w = gray.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Conservative filtering
        if h < 0.4 * img_h:
            continue
        if w < 3:
            continue

        boxes.append((x, y, w, h))

    # Sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Save characters
    for i, (x, y, w, h) in enumerate(boxes):
        char = gray[y:y+h, x:x+w]
        char_path = os.path.join(output_dir, f"char_{i}.png")
        cv2.imwrite(char_path, char)



if __name__ == "__main__":
    image_path = "image.png"
    output_root = "output"

    os.makedirs(output_root, exist_ok=True)

    word_images = extract_word_images(image_path)

    print(f"Extracted {len(word_images)} word images")

    for idx, word_img in enumerate(word_images):
        word_dir = os.path.join(output_root, f"word_{idx}")
        os.makedirs(word_dir, exist_ok=True)

        # Save word image
        word_path = os.path.join(word_dir, "word.png")
        word_img.save(word_path)

        # Extract and save characters
        extract_characters(word_img, word_dir)

