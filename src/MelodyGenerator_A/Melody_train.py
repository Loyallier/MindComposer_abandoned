import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import custom modules
from Melody_model import MelodyGPT, GPTConfig
from Melody_dataset import create_dataloaders
from Melody_tokenizer import MelodyTokenizer

# ================= Training Hyperparameters Configuration =================
# Hardware and Paths
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = os.path.join('data', 'processed', 'dataset.txt')
VOCAB_PATH = os.path.join('data', 'processed', 'Melody_vocab.json')
CKPT_PATH = os.path.join('src', 'MelodyGenerator_A', 'Melody_ckpt.pt')

# Model Parameters (Must match the design in Melody_model.py)
BLOCK_SIZE = 256      # Context window
BATCH_SIZE = 64       # RTX 4060 8GB can easily handle 64-128
N_LAYER = 6           # Number of layers
N_HEAD = 6            # Number of attention heads
N_EMBD = 384          # Embedding dimension
DROPOUT = 0.0         # 0.0 to force overfitting/memorization

# Optimizer Parameters
LEARNING_RATE = 5e-4  # Higher learning rate to accelerate convergence
MAX_ITERS = 5000      # Number of training iterations
EVAL_INTERVAL = 500   # Evaluation interval
WARMUP_STEPS = 100    # Number of warmup steps

# ==========================================================================

def estimate_loss(model, dataloaders, eval_iters=50):
    """
    Evaluation function: Calculates average Loss for training and validation sets.
    """
    out = {}
    model.eval()
    for split, loader in dataloaders.items():
        losses = torch.zeros(eval_iters)
        # Get an iterator
        loader_iter = iter(loader)
        for k in range(eval_iters):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                X, Y = next(loader_iter)
                
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    print(f"[System] Device: {DEVICE}")
    torch.manual_seed(1337) # Reproducibility

    # 1. Prepare Data
    print("[Train] Loading Data...")
    # Note: In overfitting experiments, we primarily focus on train loss.
    # For logical rigor, we still split a validation set, though val loss may rise (in pure overfitting).
    train_loader, val_loader, tokenizer = create_dataloaders(
        DATA_PATH, VOCAB_PATH, 
        block_size=BLOCK_SIZE, 
        batch_size=BATCH_SIZE
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    vocab_size = tokenizer.vocab_size
    print(f"[Train] Vocab Size: {vocab_size}")

    # 2. Initialize Model
    print("[Train] Initializing Model...")
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT
    )
    model = MelodyGPT(config)
    model.to(DEVICE)
    
    # 3. Optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print(f"[Train] Starting training for {MAX_ITERS} iterations...")
    iter_num = 0
    t0 = time.time()
    
    # Create an infinite loop iterator
    train_iter = iter(train_loader)

    while iter_num < MAX_ITERS:
        # Get Batch
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        # Evaluation Stage
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, dataloaders)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Forward Pass (Mixed Precision)
        # RTX 4060 supports AMP, significantly speeding up training
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(X, Y)

        # Backward Pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        iter_num += 1

    # 5. Finish Training and Save
    t1 = time.time()
    print(f"[Train] Finished in {(t1-t0):.2f} seconds.")
    
    print(f"[Train] Saving model to {CKPT_PATH}...")
    torch.save(model.state_dict(), CKPT_PATH)

    # 6. Generation Test (Sanity Check)
    print("-" * 30)
    print("[Train] Generating sample melody (Overfit Check)...")
    model.eval()
    
    # The Prompt can be a typical ABC file header to see if it can continue writing.
    # M:4/4
    # K:G
    start_str = "\nM:4/4\nK:G\n"
    start_ids = tokenizer.encode(start_str)
    context = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(context, max_new_tokens=200, temperature=0.8)
        
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)
    print("-" * 30)

if __name__ == "__main__":
    main()