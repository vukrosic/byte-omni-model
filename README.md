# Byte-Level Arithmetic LLM

A minimal transformer that learns arithmetic by processing raw bytes from MNIST images.

Code & Train Vision LLM in 37 Min

- YouTube - https://youtu.be/RWHKD3rI21o

- Bilibili - https://www.bilibili.com/video/BV1KTYvz3ErH/


## What it does

Learns to solve: `digit + mnist_label = result`

Example: `3 + [MNIST image of "7"] = 10`

## Quick Start

```bash
# Train the model
python llm.py

# Run inference
python inference.py

# Visualize data
python show_data.py
```

## Requirements

- PyTorch
- torchvision
- tqdm
- numpy

## How it works

1. Takes a random digit (1-9)
2. Adds it to the true label of an MNIST image
3. Learns to predict the result from raw byte sequences
4. Uses Muon optimizer for efficient training

## Files

- `llm.py` - Main training script with transformer model
- `inference.py` - Load trained model and test predictions
- `show_data.py` - Visualize training data examples

## Training

- 50,000 steps by default
- Saves best model to `best_byte_llm.pt`
- Uses mixed precision training with gradient accumulation

## Known Issues

- Model sometimes repeats digits in output (e.g., "1010" instead of "10")
- Despite formatting issues, arithmetic calculations are correct