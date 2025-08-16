#!/usr/bin/env python3
"""
Inference script for the byte-level arithmetic LLM
Allows user to input a digit and select an MNIST image to see the model's prediction
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import random
from dataclasses import dataclass
from typing import List, Optional
import os

# Import model classes from main file
from llm import MinimalLLM, ModelConfig, draw_ascii_image, draw_byte_image

def load_model(checkpoint_path: str = "best_byte_llm.pt"):
    """Load the trained model"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model file {checkpoint_path} not found!")
        print("Available files:")
        for f in os.listdir('.'):
            if f.endswith('.pt'):
                print(f"  - {f}")
        return None, None
    
    print(f"📦 Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint['config']
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Max seq len: {config.max_seq_len}")
    
    return model, config

def load_mnist_test_data():
    """Load MNIST test dataset for image selection"""
    print("📦 Loading MNIST test data...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('mnist_cache', train=False, download=True, transform=transform)
    print(f"✅ Loaded {len(test_dataset)} test images")
    return test_dataset

def create_input_sequence(digit: int, image: np.ndarray, config: ModelConfig) -> List[int]:
    """Create input sequence: digit_byte + image_bytes"""
    # Convert digit to ASCII byte
    digit_byte = ord(str(digit))
    
    # Flatten image to bytes (0-255)
    image_flat = image.flatten()
    assert len(image_flat) == 784, f"Expected 784 pixels, got {len(image_flat)}"
    image_bytes = (image_flat * 255).astype(np.uint8).tolist()
    
    # Create input sequence (no result yet)
    sequence = [digit_byte]
    sequence.extend(image_bytes)
    
    return sequence

def predict_result(model, input_sequence: List[int], config: ModelConfig, max_new_tokens: int = 3):
    """Generate the result using the model"""
    device = next(model.parameters()).device
    
    # Pad input to match training format
    padded_input = input_sequence.copy()
    while len(padded_input) < config.max_seq_len:
        padded_input.append(config.PAD_TOKEN)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(padded_input[:-max_new_tokens], dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"🤖 Generating result...")
    print(f"   Input length: {len(input_sequence)} bytes")
    print(f"   Padded length: {len(padded_input)} bytes")
    
    generated = input_tensor.clone()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Get model predictions
            logits = model(generated)
            
            # Get next token (last position)
            next_token_logits = logits[0, -1, :]
            
            # Show top predictions for debugging
            top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), 5)
            print(f"   Top 5 predictions:")
            for j in range(5):
                token_val = top_indices[j].item()
                prob = top_probs[j].item()
                char_repr = f"'{chr(token_val)}'" if 32 <= token_val <= 126 else f"byte_{token_val}"
                print(f"     {token_val:3d} ({char_repr}): {prob:.3f}")
            
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Add to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            print(f"   Generated token {i+1}: {next_token.item()}", end="")
            if 48 <= next_token.item() <= 57:
                print(f" = '{chr(next_token.item())}'")
            else:
                print(f" (not a digit)")
            
            # Stop if we get padding token or after generating result
            if next_token.item() == config.PAD_TOKEN:
                print(f"   Stopped at padding token")
                break
    
    # Extract generated result
    result_tokens = generated[0, len(input_tensor[0]):].cpu().tolist()
    result_str = ''.join([chr(t) for t in result_tokens if 48 <= t <= 57])
    
    return result_str, result_tokens

def show_prediction_details(digit: int, image: np.ndarray, true_label: int, predicted_result: str, result_tokens: List[int]):
    """Show detailed prediction results"""
    print(f"\n🔍 PREDICTION DETAILS:")
    print("=" * 80)
    
    print(f"📝 Input digit: {digit}")
    print(f"🏷️  True MNIST label: {true_label}")
    print(f"🧮 Expected result: {digit} + {true_label} = {digit + true_label}")
    print(f"🤖 Model prediction: '{predicted_result}'")
    print(f"🔢 Raw result tokens: {result_tokens}")
    
    # Show the image
    image_bytes = (image.flatten() * 255).astype(np.uint8).tolist()
    draw_ascii_image(image_bytes)
    print()
    draw_byte_image(image_bytes)
    
    # Verify result
    try:
        predicted_num = int(predicted_result) if predicted_result else -1
        expected_num = digit + true_label
        correct = predicted_num == expected_num
        
        print(f"\n✅ Correct: {correct}")
        if not correct:
            print(f"   Expected: {expected_num}")
            print(f"   Got: {predicted_num}")
    except:
        print(f"\n❌ Could not parse prediction as number")
    
    print("=" * 80)

def interactive_mode(model, config: ModelConfig, test_dataset):
    """Interactive mode for testing the model"""
    print(f"\n🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Commands:")
    print("  - Enter a digit (1-9) to use as first operand")
    print("  - 'random' - use random MNIST image")
    print("  - 'select N' - use MNIST image at index N")
    print("  - 'quit' - exit")
    print("=" * 50)
    
    while True:
        try:
            # Get user input for digit
            user_input = input(f"\n🔢 Enter digit (1-9) or command: ").strip().lower()
            
            if user_input == 'quit':
                break
            
            # Parse digit
            try:
                digit = int(user_input)
                if not (1 <= digit <= 9):
                    print("❌ Please enter a digit between 1-9")
                    continue
            except ValueError:
                print("❌ Please enter a valid digit or command")
                continue
            
            # Get image selection
            img_input = input(f"🖼️  Image selection ('random' or index 0-{len(test_dataset)-1}): ").strip().lower()
            
            if img_input == 'random':
                img_idx = random.randint(0, len(test_dataset) - 1)
            else:
                try:
                    img_idx = int(img_input)
                    if not (0 <= img_idx < len(test_dataset)):
                        print(f"❌ Index must be between 0 and {len(test_dataset)-1}")
                        continue
                except ValueError:
                    print("❌ Please enter 'random' or a valid index")
                    continue
            
            # Get the image
            image, true_label = test_dataset[img_idx]
            image_np = image.squeeze().numpy()
            
            print(f"\n🎯 Testing: {digit} + MNIST[{img_idx}] (label={true_label})")
            
            # Create input sequence
            input_seq = create_input_sequence(digit, image_np, config)
            
            # Get prediction
            predicted_result, result_tokens = predict_result(model, input_seq, config)
            
            # Show results
            show_prediction_details(digit, image_np, true_label, predicted_result, result_tokens)
            
        except KeyboardInterrupt:
            print(f"\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_test_mode(model, config: ModelConfig, test_dataset, num_tests: int = 10):
    """Test the model on random examples"""
    print(f"\n🧪 BATCH TEST MODE - {num_tests} random examples")
    print("=" * 80)
    
    correct = 0
    total = 0
    
    for i in range(num_tests):
        # Random digit and image
        digit = random.randint(1, 9)
        img_idx = random.randint(0, len(test_dataset) - 1)
        image, true_label = test_dataset[img_idx]
        image_np = image.squeeze().numpy()
        
        # Create input and predict
        input_seq = create_input_sequence(digit, image_np, config)
        predicted_result, result_tokens = predict_result(model, input_seq, config)
        
        # Check correctness
        expected = digit + true_label
        try:
            predicted_num = int(predicted_result) if predicted_result else -1
            is_correct = predicted_num == expected
            correct += is_correct
        except:
            is_correct = False
        
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Test {i+1}: {digit} + {true_label} = {expected}, predicted: '{predicted_result}'")
    
    accuracy = correct / total * 100
    print(f"\n📊 Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")

def debug_model_predictions(model, config: ModelConfig, test_dataset):
    """Debug what the model is actually predicting"""
    print(f"\n🔍 DEBUG MODE - Analyzing model behavior")
    print("=" * 80)
    
    device = next(model.parameters()).device
    
    # Test with a simple known example
    digit = 5
    img_idx = 0
    image, true_label = test_dataset[img_idx]
    image_np = image.squeeze().numpy()
    
    print(f"🧪 Debug example: {digit} + {true_label} = {digit + true_label}")
    
    # Create full training-style sequence
    input_seq = create_input_sequence(digit, image_np, config)
    expected_result = digit + true_label
    expected_result_bytes = [ord(c) for c in str(expected_result)]
    
    # Create full sequence as model was trained
    full_sequence = input_seq + expected_result_bytes
    while len(full_sequence) < config.max_seq_len:
        full_sequence.append(config.PAD_TOKEN)
    
    print(f"📏 Full sequence length: {len(full_sequence)}")
    print(f"🎯 Expected result bytes: {expected_result_bytes} = '{str(expected_result)}'")
    
    # Test model on the full sequence
    input_tensor = torch.tensor(full_sequence[:-1], dtype=torch.long).unsqueeze(0).to(device)
    target_tensor = torch.tensor(full_sequence[1:], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1)
        
        # Check predictions at result positions
        result_start = len(input_seq)
        result_positions = list(range(result_start, result_start + len(expected_result_bytes)))
        
        print(f"\n🔍 Model predictions at result positions:")
        for i, pos in enumerate(result_positions):
            if pos < logits.size(1):
                pred_token = predictions[0, pos].item()
                target_token = target_tensor[0, pos].item()
                
                pred_char = chr(pred_token) if 48 <= pred_token <= 57 else f"byte_{pred_token}"
                target_char = chr(target_token) if 48 <= target_token <= 57 else f"byte_{target_token}"
                
                match = "✅" if pred_token == target_token else "❌"
                print(f"   Position {pos}: predicted {pred_token} ({pred_char}), target {target_token} ({target_char}) {match}")
        
        # Check overall accuracy on this sequence
        correct = (predictions == target_tensor).float().mean().item()
        print(f"\n📊 Sequence accuracy: {correct:.3f}")

def main():
    print("🚀 Byte-Level Arithmetic LLM Inference")
    print("=" * 50)
    
    # Load model
    model, config = load_model()
    if model is None:
        return
    
    # Load test data
    test_dataset = load_mnist_test_data()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🔧 Using device: {device}")
    
    while True:
        print(f"\n📋 MAIN MENU")
        print("1. Interactive mode (manual input)")
        print("2. Batch test mode (random examples)")
        print("3. Debug mode (analyze model behavior)")
        print("4. Quit")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            interactive_mode(model, config, test_dataset)
        elif choice == '2':
            num_tests = input("Number of tests (default 10): ").strip()
            try:
                num_tests = int(num_tests) if num_tests else 10
            except:
                num_tests = 10
            batch_test_mode(model, config, test_dataset, num_tests)
        elif choice == '3':
            debug_model_predictions(model, config, test_dataset)
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()