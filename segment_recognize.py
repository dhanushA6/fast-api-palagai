# segment_recognize.py
from base64 import b64decode
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage
import torch
import cv2
import json
import unicodedata
import inference

def word_to_img(dataURL):
    """Convert data URL to image."""
    string = str(dataURL)
    comma = string.find(",")
    code = string[comma + 1:]
    decoded = b64decode(code)
    buf = BytesIO(decoded)
    img = Image.open(buf)
    
    # Convert to grayscale with white background
    converted = img.convert("LA")
    la = np.array(converted)
    la[la[..., -1] == 0] = [255, 255]
    whiteBG = Image.fromarray(la)
    
    # Convert to binary image
    converted = whiteBG.convert("L")
    inverted = ImageOps.invert(converted)
    
    return inverted

def segment_characters(img):
    """Segment the word image into individual characters."""
    # Convert PIL image to OpenCV format
    img_np = np.array(img)
    
    # Threshold the image to create a binary image
    _, binary = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean the image
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    # Analyze connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter out small noise components and the background (labeled as 0)
    char_components = []
    for i in range(1, num_labels):  # Skip background (0)
        x, y, w, h, area = stats[i]
        
        # Filter small noise components
        if area > 50:  # Adjust this threshold based on your data
            char_components.append((x, y, w, h, i))
    
    # Sort components from right to left (Tamil is written left to right, but some characters might be composed)
    char_components.sort(key=lambda x: x[0])
    
    # Group closely spaced components that might form a single character
    merged_components = []
    if len(char_components) > 0:
        current_group = [char_components[0]]
        
        for i in range(1, len(char_components)):
            current_comp = char_components[i]
            prev_comp = current_group[-1]
            
            # Calculate horizontal distance between components
            distance = current_comp[0] - (prev_comp[0] + prev_comp[2])
            
            # If components are close, add to current group
            # Tamil characters often have multiple components
            if distance < 0.01:  # Adjust this threshold based on your data
                current_group.append(current_comp)
            else:
                # Process current group
                x_min = min(comp[0] for comp in current_group)
                y_min = min(comp[1] for comp in current_group)
                x_max = max(comp[0] + comp[2] for comp in current_group)
                y_max = max(comp[1] + comp[3] for comp in current_group)
                merged_components.append((x_min, y_min, x_max - x_min, y_max - y_min))
                
                # Start new group
                current_group = [current_comp]
        
        # Process the last group
        if current_group:
            x_min = min(comp[0] for comp in current_group)
            y_min = min(comp[1] for comp in current_group)
            x_max = max(comp[0] + comp[2] for comp in current_group)
            y_max = max(comp[1] + comp[3] for comp in current_group)
            merged_components.append((x_min, y_min, x_max - x_min, y_max - y_min))
    
    # Extract and preprocess each character for recognition
    character_images = []
    for x, y, w, h in merged_components:
        # Add padding around the character
        padding = 5
        y_start = max(0, y - padding)
        y_end = min(binary.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(binary.shape[1], x + w + padding)
        
        # Extract character region with padding
        char_img = binary[y_start:y_end, x_start:x_end]
        
        # Convert to PIL image
        char_pil = Image.fromarray(char_img)
        
        # Apply same preprocessing as in the inference module
        thick = char_pil.filter(ImageFilter.MaxFilter(5))
        
        # Resize to fit the model input requirements
        ratio = 48.0 / max(thick.size)
        new_size = tuple([int(round(x*ratio)) for x in thick.size])
        resized = thick.resize(new_size, Image.LANCZOS)
        
        # Center the character
        arr = np.asarray(resized)
        com = ndimage.measurements.center_of_mass(arr)
        result = Image.new("L", (64, 64))
        box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
        result.paste(resized, box)
        
        character_images.append(result)
    
    return character_images

import unicodedata

def fix_left_side_vowels(chars):
    """Fix Tamil left-side vowels like 'ெ', 'ே', 'ை' appearing before consonants."""
    left_side_vowels = {'ெ', 'ே', 'ை'}
    consonant_range = ('க', 'ஹ')  # Tamil consonants from க to ஹ
    
    fixed_chars = []
    i = 0
    while i < len(chars):
        if chars[i] in left_side_vowels:
            if i + 1 < len(chars) and consonant_range[0] <= chars[i + 1] <= consonant_range[1]:
                # Swap vowel and consonant
                fixed_chars.append(chars[i + 1])
                fixed_chars.append(chars[i])
                i += 2
            else:
                fixed_chars.append(chars[i])
                i += 1
        else:
            fixed_chars.append(chars[i])
            i += 1
    return fixed_chars

def recognize_word(dataURL, net):
    """Segment word into characters and recognize each character."""
    # Convert data URL to image
    word_img = word_to_img(dataURL)
    
    # Segment into characters
    char_images = segment_characters(word_img)
    
    recognized_chars = []
    confidences = []
    
    for img in char_images:
        # Transform the image
        transformed = inference.transformImg(img)
        
        # Get prediction
        output = net(transformed)
        prob, predicted = torch.max(output.data, 1)
        confidence = int(round(prob.item() * 100))
        
        # Get the predicted character and confidence
        char_idx = predicted.item()
        recognized_chars.append(inference.classes[char_idx])
        confidences.append(confidence)
    
    # 🔥 Fix left-side vowel placement
    recognized_chars = fix_left_side_vowels(recognized_chars)
    
    # Combine characters into word
    recognized_word = ''.join(recognized_chars)
    
    # Normalize Unicode for proper combination
    recognized_word = unicodedata.normalize('NFC', recognized_word)
    
    # Calculate average confidence
    avg_confidence = int(sum(confidences) / len(confidences)) if confidences else 0
    
    return {
        'word': recognized_word,
        'confidence': avg_confidence,
        'characters': [{'char': char, 'confidence': conf} for char, conf in zip(recognized_chars, confidences)]
    }

