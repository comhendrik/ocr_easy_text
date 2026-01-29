import cv2
import numpy as np

def segment_characters(image_path):
    # Bild laden und vorverarbeiten
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarisierung (Otsu's Methode)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Konturen finden (einzelne Buchstaben)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Bounding Boxes extrahieren und sortieren (links nach rechts)
    char_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and w > 5:  # Filter für Mindestgröße
            char_boxes.append((x, y, w, h))
    
    char_boxes.sort(key=lambda b: b[0])  # Nach x-Position sortieren
    
    # Buchstaben ausschneiden
    characters = []
    for (x, y, w, h) in char_boxes:
        char_img = gray[y:y+h, x:x+w]
        # Auf einheitliche Größe bringen (z.B. 32x32)
        char_img = cv2.resize(char_img, (32, 32))
        characters.append(char_img)
    
    return characters, char_boxes