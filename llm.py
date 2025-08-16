import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import os
import pickle
from torchvision import datasets, transforms
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 16
    max_steps: int = 5000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters - byte-level
    max_seq_len: int = 787  # 1 digit + 784 image + 2 result = 787 bytes max
    vocab_size: int = 256  # 0-255 byte values
    num_samples: int = 10000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True

    # No special tokens needed - just raw bytes
    PAD_TOKEN: int = 255  # Use 255 for padding

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_mnist_data(cache_dir: str = "mnist_cache"):
    """Load MNIST dataset and cache it"""
    os.makedirs(cache_dir, exist_ok=True)
    
    print("📦 Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(cache_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(cache_dir, train=False, download=True, transform=transform)
    
    print(f"✅ Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_dataset, test_dataset

def create_addition_sequence(digit1: int, image2: np.ndarray, true_label: int, config: ModelConfig) -> List[int]:
    """Create a byte sequence for: digit1_byte + image_bytes = result_bytes"""
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
    
    # Simple format: digit_byte + image_bytes + result_bytes
    sequence = [digit1_byte]  # First character as text byte
    sequence.extend(image_bytes)  # Full raw image bytes
    sequence.extend(result_bytes)  # Result as 1-2 text bytes
    
    return sequence

def generate_arithmetic_data(mnist_train, mnist_test, config: ModelConfig):
    """Generate arithmetic sequences from MNIST data"""
    print("🔢 Generating arithmetic sequences...")
    
    sequences = []
    
    # Use subset of MNIST for faster training
    indices = random.sample(range(len(mnist_train)), min(config.num_samples, len(mnist_train)))
    
    for idx in tqdm(indices, desc="Creating sequences"):
        image, label = mnist_train[idx]
        image_np = image.squeeze().numpy()
        
        # Random digit 1-9 for first operand
        digit1 = random.randint(1, 9)
        
        # Create sequence
        sequence = create_addition_sequence(digit1, image_np, label, config)
        
        # Pad to fixed length for batching
        while len(sequence) < config.max_seq_len:
            sequence.append(config.PAD_TOKEN)
        
        sequences.append(sequence)
    
    print(f"✅ Generated {len(sequences)} arithmetic sequences")
    return sequences

def draw_ascii_image(image_bytes: List[int], width: int = 28, height: int = 28):
    """Draw image using ASCII characters based on pixel intensity"""
    if len(image_bytes) != width * height:
        print(f"Warning: Expected {width*height} pixels, got {len(image_bytes)}")
        return
    
    # ASCII characters from dark to light (these are just visual representations)
    chars = " .:-=+*#%@"
    
    print("📸 MNIST Image - ASCII Art:")
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

def draw_byte_image(image_bytes: List[int], width: int = 28, height: int = 28):
    """Draw image showing actual byte values"""
    if len(image_bytes) != width * height:
        print(f"Warning: Expected {width*height} pixels, got {len(image_bytes)}")
        return
    
    print("🔢 Raw Bytes (first 5 rows):")
    for row in range(min(5, height)):
        line = f"Row {row:2d}: "
        for col in range(width):
            pixel_val = image_bytes[row * width + col]
            line += f"{pixel_val:3d} "
        print(line)
    
    if height > 5:
        print(f"... ({height - 5} more rows)")

def show_training_example(input_seq, target_seq, pred_seq, step):
    """Show a training example during training"""
    print(f"\n🔍 TRAINING EXAMPLE AT STEP {step}")
    print("=" * 80)
    
    # Convert tensors to lists
    input_bytes = input_seq.cpu().tolist()
    target_bytes = target_seq.cpu().tolist()
    pred_bytes = pred_seq.cpu().tolist()
    
    # Parse the sequence: digit + image + result
    digit_byte = input_bytes[0]
    image_bytes = input_bytes[1:785]  # 784 bytes
    
    # Find where result starts in target (after image)
    result_start = 785
    target_result = target_bytes[result_start:]
    pred_result = pred_bytes[result_start:]
    
    print(f"📝 Input digit: {digit_byte} = '{chr(digit_byte) if 48 <= digit_byte <= 57 else '?'}'")
    print(f"📊 Sequence length: {len(input_bytes)}")
    
    # Show the image
    draw_ascii_image(image_bytes)
    print()
    draw_byte_image(image_bytes)
    
    # Show results
    target_str = ''.join([chr(b) for b in target_result if 48 <= b <= 57])
    pred_str = ''.join([chr(b) for b in pred_result if 48 <= b <= 57])
    
    print(f"\n🎯 Target result: {target_result[:5]} -> '{target_str}'")
    print(f"🤖 Predicted result: {pred_result[:5]} -> '{pred_str}'")
    print(f"✅ Match: {target_str == pred_str}")
    
    print("=" * 80)

def visualize_training_data(sequences: List[List[int]], config: ModelConfig, num_examples: int = 3):
    """Visualize what the training data looks like"""
    print(f"\n🔍 TRAINING DATA EXAMPLES:")
    print("=" * 80)
    
    for i in range(min(num_examples, len(sequences))):
        sequence = sequences[i]
        print(f"\n📊 Example {i+1}:")
        print("-" * 60)
        
        # Parse simple format: digit + image + result
        digit_byte = sequence[0]
        image_bytes = sequence[1:785]  # 784 bytes
        result_bytes = sequence[785:]  # 1-2 bytes
        
        digit_char = chr(digit_byte) if 48 <= digit_byte <= 57 else '?'
        result_str = ''.join([chr(b) for b in result_bytes if 48 <= b <= 57])
        
        print(f"🧮 Format: '{digit_char}' + [MNIST image] = '{result_str}'")
        print(f"📏 Sequence length: {len(sequence)} bytes")
        print(f"🎯 Structure: DIGIT(0) → IMAGE(1:785) → RESULT(785:)")
        
        # Draw the MNIST image
        if len(image_bytes) == 784:
            draw_ascii_image(image_bytes)
            print()
            draw_byte_image(image_bytes)
        else:
            print(f"⚠️  Image has wrong size: {len(image_bytes)} bytes")
        
        # Show statistics
        print(f"📊 Image stats: min={min(image_bytes)}, max={max(image_bytes)}, mean={np.mean(image_bytes):.1f}, non-zero={sum(1 for b in image_bytes if b > 0)}")
        
        # Show key bytes
        print(f"🔢 Digit byte: {digit_byte} = '{digit_char}'")
        print(f"🔢 Result bytes: {result_bytes} = '{result_str}'")
        print(f"🔢 First 10 image bytes: {image_bytes[:10]}")
        
        # Verify arithmetic
        try:
            digit_val = int(digit_char)
            result_val = int(result_str)
            expected_second = result_val - digit_val
            print(f"✅ Arithmetic: {digit_val} + {expected_second} = {result_val}")
        except:
            print(f"❌ Could not parse arithmetic")
        
        print()
    
    print("=" * 80)

class ByteArithmeticDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_seq_len: int, pad_token: int):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        
        # Ensure sequence is exactly max_seq_len
        if len(sequence) < self.max_seq_len:
            sequence.extend([self.pad_token] * (self.max_seq_len - len(sequence)))
        elif len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        
        # Input is sequence[:-1], target is sequence[1:]
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with Muon optimizer"""
    print(f"\n🚀 Training Byte-Level Arithmetic LLM with Muon optimizer")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging and visualization
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })
                
                # Show training example every 500 steps
                if step % 500 == 0 and step > 0:
                    show_training_example(x[0], y[0], predictions[0], step)

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'eval_metrics': eval_metrics
                    }, 'best_byte_llm.pt')
                    print(f"💾 Saved best model (val_loss: {best_val_loss:.4f})")

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.num_samples:,} samples, seq_len {config.max_seq_len}")
    print(f"   Vocabulary: {config.vocab_size} bytes (0-255)")

    # Load MNIST data
    mnist_train, mnist_test = load_mnist_data()
    
    # Generate arithmetic sequences
    sequences = generate_arithmetic_data(mnist_train, mnist_test, config)
    
    # Show examples of training data
    visualize_training_data(sequences, config, num_examples=3)
    
    dataset = ByteArithmeticDataset(sequences, config.max_seq_len, config.PAD_TOKEN)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"🔢 Byte vocabulary size: {config.vocab_size}")
    print(f"📏 Max sequence length: {config.max_seq_len}")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_metrics,
        'total_time': total_time
    }, 'final_byte_llm.pt')

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"💾 Models saved: best_byte_llm.pt, final_byte_llm.pt")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")