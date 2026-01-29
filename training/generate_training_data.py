from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import random

class PrintedGermanCharGenerator:
    """Generator für gedruckte deutsche Zeichen"""

    GERMAN_CHARS = (
        list("abcdefghijklmnopqrstuvwxyz") +
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
        list("äöüÄÖÜß") +
        list("0123456789") +
        list(".,!?-:;")
    )

    STANDARD_FONTS = [
        "arial.ttf",
        "calibri.ttf",
        "times.ttf",
        "verdana.ttf",
        "tahoma.ttf",
        "segoeui.ttf",
        "trebuc.ttf",
        "georgia.ttf",
        "cour.ttf",
        "consola.ttf",
        "candara.ttf",
        "constan.ttf",
        "corbel.ttf",
        "micross.ttf",
        "lucon.ttf",
        "pala.ttf",
        "impact.ttf",
        "framd.ttf",
    ]


    def __init__(self, img_size=32):
        self.img_size = img_size
        self.fonts = self._load_fonts()
        self.char_to_idx = {c: i for i, c in enumerate(self.GERMAN_CHARS)}
        self.idx_to_char = {i: c for i, c in enumerate(self.GERMAN_CHARS)}

        print(f"{len(self.fonts)} Fonts geladen")
        print(f"{len(self.GERMAN_CHARS)} Zeichen im Alphabet")
    
    def _load_fonts(self):
        fonts = []
        
        # Windows Font-Verzeichnis
        font_dir = "C:/Windows/Fonts/"
        
        if os.path.exists(font_dir):
            for font_name in self.STANDARD_FONTS:
                path = os.path.join(font_dir, font_name)
                if os.path.exists(path):
                    fonts.append(path)
                    print(f"  [OK] {font_name}")
                else:
                    print(f"  [--] {font_name} nicht gefunden")
        
        if not fonts:
            print("Keine Fonts gefunden!")
        
        return fonts
    
    def generate_char(self, char, augment=True):
        """Einzelnes Zeichen generieren"""
        
        font_path = random.choice(self.fonts)
        font_size = random.randint(18, 26) if augment else 22
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        bg = random.randint(235, 255) if augment else 255
        fg = random.randint(0, 40) if augment else 0
        
        img = Image.new('L', (self.img_size, self.img_size), bg)
        draw = ImageDraw.Draw(img)
        
        bbox = draw.textbbox((0, 0), char, font=font)
        x = (self.img_size - bbox[2] + bbox[0]) // 2 - bbox[0]
        y = (self.img_size - bbox[3] + bbox[1]) // 2 - bbox[1]
        
        if augment:
            x += random.randint(-2, 2)
            y += random.randint(-2, 2)
        
        draw.text((x, y), char, font=font, fill=fg)
        
        if augment:
            img = self._simulate_print(img)
        
        return np.array(img, dtype=np.float32) / 255.0
    
    def _simulate_print(self, img):
        """Simuliert Scan/Foto von gedrucktem Text"""
        
        if random.random() > 0.6:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        if random.random() > 0.5:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 3, arr.shape)
            arr = np.clip(arr + noise, 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))
        
        return img
    
    def generate_dataset(self, samples_per_char=300):
        """Kompletten Datensatz generieren"""
        
        images, labels = [], []
        total = len(self.GERMAN_CHARS) * samples_per_char
        
        print(f"Generiere {total} Samples...")
        
        for i, char in enumerate(self.GERMAN_CHARS):
            for _ in range(samples_per_char):
                img = self.generate_char(char, augment=True)
                images.append(img)
                labels.append(i)
            
            print(f"\r  {i + 1}/{len(self.GERMAN_CHARS)} Zeichen", end="")
        
        print("\n  Mische Datensatz...")
        
        indices = np.random.permutation(len(images))
        images = np.array(images)[indices]
        labels = np.array(labels)[indices]
        
        return images, labels
    
    @property
    def num_classes(self):
        return len(self.GERMAN_CHARS)


# Ausführen
gen = PrintedGermanCharGenerator()

X, y = gen.generate_dataset(samples_per_char=300)
print(f"Dataset: {X.shape}")

np.savez_compressed("training/german_printed_chars.npz", X=X, y=y, chars=gen.GERMAN_CHARS)
print("Gespeichert: german_printed_chars.npz")