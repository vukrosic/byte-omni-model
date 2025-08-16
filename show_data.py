#!/usr/bin/env python3
"""
Quick script to show what the training data looks like
"""
import torch
import numpy as np
import random
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    # Special tokens
    PLUS_TOKEN: int = 250  # '+' 
    EQUALS_TOKEN: int = 251  # '='
    PAD_TOKEN: int = 252
    START_TOKEN: int = 253
    END_TOKEN: int = 254
    max_seq_len: int = 1600

def load_mnist_sample():
    """Load a single MNIST sample"""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('mnist_cache', train=True, download=True, transform=transform)
    
    # Get a random sample
    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]
    return image.squeeze().numpy(), label

def create_addition_sequence(digit1: int, image2: np.ndarray, true_label: int, config: ModelConfig) -> List[int]:
    """Create a byte sequence for: digit1 + image_label = result"""
    # Convert digit to ASCII byte (48-57 for '0'-'9')
    digit1_byte = ord(str(digit1))
    
    # Flatten image to bytes (0-255) - ensure exactly 784 bytes
    image_flat = image2.flatten()
    assert len(image_flat) == 784, f"Expected 784 pixels, got {len(image_flat)}"
    image_bytes = (image_flat * 255).astype(np.uint8).tolist()
    
    # Use the true MNIST label as second operand
    result = digit1 + true_label
    result_str = str(result)
    result_bytes = [ord(c) for c in result_str]
    
    print(f"  Creating: {digit1} + {true_label} = {result}")
    print(f"  Digit byte: {digit1_byte} ('{chr(digit1_byte)}')")
    print(f"  Image bytes: {len(image_bytes)} pixels")
    print(f"  Result bytes: {result_bytes} -> '{result_str}'")
    
    # Create sequence: START + digit1 + PLUS + image_bytes + EQUALS + result + END
    sequence = [config.START_TOKEN]
    sequence.append(digit1_byte)
    sequence.append(config.PLUS_TOKEN)
    sequence.extend(image_bytes)
    sequence.append(config.EQUALS_TOKEN)
    sequence.extend(result_bytes)
    sequence.append(config.END_TOKEN)
    
    return sequence, true_label, result

def draw_ascii_image(image_bytes: List[int], width: int = 28, height: int = 28):
    """Draw image using ASCII characters based on pixel intensity"""
    if len(image_bytes) != width * height:
        print(f"Warning: Expected {width*height} pixels, got {len(image_bytes)}")
        return
    
    # ASCII characters from dark to light
    chars = " .:-=+*#%@"
    
    print("📸 MNIST Image (28x28 pixels):")
    print("+" + "-" * width + "+")
    
    for row in range(height):
        line = "|"
        for col in range(width):
            pixel_val = image_bytes[row * width + col]
            # Map 0-255 to 0-9 (length of chars - 1)
            char_idx = min(9, pixel_val * 9 // 255)
            line += chars[char_idx]
        line += "|"
        print(line)
    
    print("+" + "-" * width + "+")

def visualize_sequence(sequence: List[int], config: ModelConfig, digit1: int, mnist_label: int, result: int):
    """Show detailed breakdown of a sequence"""
    print(f"\n🔍 SEQUENCE BREAKDOWN:")
    print("=" * 80)
    
    # Find positions
    start_pos = sequence.index(config.START_TOKEN)
    plus_pos = sequence.index(config.PLUS_TOKEN)
    equals_pos = sequence.index(config.EQUALS_TOKEN)
    end_pos = sequence.index(config.END_TOKEN)
    
    print(f"🧮 Operation: {digit1} + {mnist_label} = {result}")
    print(f"📏 Total sequence length: {len(sequence)} bytes")
    print()
    
    # Show structure
    print("📋 Sequence structure:")
    print(f"  [{start_pos:3d}] START_TOKEN ({config.START_TOKEN})")
    print(f"  [{start_pos+1:3d}] Digit byte: {sequence[start_pos+1]} = '{chr(sequence[start_pos+1])}'")
    print(f"  [{plus_pos:3d}] PLUS_TOKEN ({config.PLUS_TOKEN})")
    print(f"  [{plus_pos+1:3d}:{equals_pos:3d}] Image bytes: {equals_pos - plus_pos - 1} bytes (28x28 = 784)")
    print(f"  [{equals_pos:3d}] EQUALS_TOKEN ({config.EQUALS_TOKEN})")
    print(f"  [{equals_pos+1:3d}:{end_pos:3d}] Result bytes: {[chr(b) for b in sequence[equals_pos+1:end_pos]]}")
    print(f"  [{end_pos:3d}] END_TOKEN ({config.END_TOKEN})")
    print()
    
    # Extract and visualize the image
    image_bytes = sequence[plus_pos+1:equals_pos]
    if len(image_bytes) == 784:  # 28x28
        draw_ascii_image(image_bytes)
    else:
        print(f"⚠️  Image has wrong size: {len(image_bytes)} bytes (expected 784)")
    
    print()
    
    # Show image statistics
    print(f"📊 Image statistics:")
    print(f"  Min pixel: {min(image_bytes)}")
    print(f"  Max pixel: {max(image_bytes)}")
    print(f"  Mean pixel: {np.mean(image_bytes):.1f}")
    print(f"  Non-zero pixels: {sum(1 for b in image_bytes if b > 0)}")
    print()
    
    # Show first 30 bytes with interpretation
    print("🔢 First 30 bytes of sequence:")
    for i in range(min(30, len(sequence))):
        byte_val = sequence[i]
        if byte_val == config.START_TOKEN:
            interpretation = "START"
        elif byte_val == config.PLUS_TOKEN:
            interpretation = "PLUS"
        elif byte_val == config.EQUALS_TOKEN:
            interpretation = "EQUALS"
        elif byte_val == config.END_TOKEN:
            interpretation = "END"
        elif 48 <= byte_val <= 57:
            interpretation = f"digit '{chr(byte_val)}'"
        else:
            interpretation = f"pixel {byte_val}"
        
        print(f"  [{i:2d}] {byte_val:3d} -> {interpretation}")
    
    if len(sequence) > 30:
        print(f"  ... ({len(sequence) - 30} more bytes)")
    
    print()
    
    # Show the actual arithmetic in the sequence
    print("🎯 Arithmetic verification:")
    digit_char = chr(sequence[start_pos + 1])
    result_chars = ''.join([chr(b) for b in sequence[equals_pos+1:end_pos]])
    print(f"  Sequence shows: '{digit_char}' + [image] = '{result_chars}'")
    print(f"  Expected: {digit1} + {mnist_label} = {result}")
    print(f"  ✅ Correct!" if result_chars == str(result) else f"  ❌ Mismatch!")

if __name__ == "__main__":
    print("🔢 MNIST Byte Arithmetic Data Visualization")
    print("=" * 60)
    
    config = ModelConfig()
    
    # Generate a few examples
    for i in range(3):
        print(f"\n📊 EXAMPLE {i+1}")
        
        # Load MNIST sample
        image, true_label = load_mnist_sample()
        
        # Random first digit
        digit1 = random.randint(1, 9)
        
        # Create sequence
        sequence, mnist_label, result = create_addition_sequence(digit1, image, true_label, config)
        
        print(f"MNIST true label: {true_label}")
        print(f"Using label as second operand: {mnist_label}")
        
        # Visualize
        visualize_sequence(sequence, config, digit1, mnist_label, result)
        
        print("\n" + "="*60)