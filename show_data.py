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

def create_addition_sequence(digit1: int, image2: np.ndarray, config: ModelConfig) -> List[int]:
    """Create a byte sequence for: digit1 + image2_bytes = result"""
    # Convert digit to ASCII byte (48-57 for '0'-'9')
    digit1_byte = ord(str(digit1))
    
    # Flatten image to bytes (0-255)
    image_bytes = (image2.flatten() * 255).astype(np.uint8).tolist()
    
    # Calculate result (simple: digit + MNIST label)
    # For demo, let's use the actual MNIST label as second operand
    mnist_label = np.argmax(np.bincount((image2.flatten() * 10).astype(int))) % 10
    result = digit1 + mnist_label
    result_str = str(result)
    result_bytes = [ord(c) for c in result_str]
    
    # Create sequence: START + digit1 + PLUS + image_bytes + EQUALS + result + END
    sequence = [config.START_TOKEN]
    sequence.append(digit1_byte)
    sequence.append(config.PLUS_TOKEN)
    sequence.extend(image_bytes)
    sequence.append(config.EQUALS_TOKEN)
    sequence.extend(result_bytes)
    sequence.append(config.END_TOKEN)
    
    return sequence, mnist_label, result

def visualize_sequence(sequence: List[int], config: ModelConfig, digit1: int, mnist_label: int, result: int):
    """Show detailed breakdown of a sequence"""
    print(f"\n🔍 SEQUENCE BREAKDOWN:")
    print("=" * 60)
    
    # Find positions
    start_pos = sequence.index(config.START_TOKEN)
    plus_pos = sequence.index(config.PLUS_TOKEN)
    equals_pos = sequence.index(config.EQUALS_TOKEN)
    end_pos = sequence.index(config.END_TOKEN)
    
    print(f"Operation: {digit1} + {mnist_label} = {result}")
    print(f"Total sequence length: {len(sequence)} bytes")
    print()
    
    # Show structure
    print("Sequence structure:")
    print(f"  [{start_pos}] START_TOKEN ({config.START_TOKEN})")
    print(f"  [{start_pos+1}] Digit byte: {sequence[start_pos+1]} = '{chr(sequence[start_pos+1])}'")
    print(f"  [{plus_pos}] PLUS_TOKEN ({config.PLUS_TOKEN})")
    print(f"  [{plus_pos+1}:{equals_pos}] Image bytes: {equals_pos - plus_pos - 1} bytes (28x28 = 784)")
    print(f"  [{equals_pos}] EQUALS_TOKEN ({config.EQUALS_TOKEN})")
    print(f"  [{equals_pos+1}:{end_pos}] Result bytes: {[chr(b) for b in sequence[equals_pos+1:end_pos]]}")
    print(f"  [{end_pos}] END_TOKEN ({config.END_TOKEN})")
    print()
    
    # Show first 20 bytes with interpretation
    print("First 20 bytes:")
    for i in range(min(20, len(sequence))):
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
    
    # Show image statistics
    image_bytes = sequence[plus_pos+1:equals_pos]
    print(f"\nImage statistics:")
    print(f"  Min pixel: {min(image_bytes)}")
    print(f"  Max pixel: {max(image_bytes)}")
    print(f"  Mean pixel: {np.mean(image_bytes):.1f}")
    print(f"  Non-zero pixels: {sum(1 for b in image_bytes if b > 0)}")

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
        sequence, mnist_label, result = create_addition_sequence(digit1, image, config)
        
        print(f"MNIST true label: {true_label}")
        print(f"Computed label from pixels: {mnist_label}")
        
        # Visualize
        visualize_sequence(sequence, config, digit1, mnist_label, result)
        
        print("\n" + "="*60)