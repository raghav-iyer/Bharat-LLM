import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model.transformer_decoder import TransformerDecoder
from model.tokenizer.tokenizer_utils import load_tokenizer
from training.dataset import MultilingualPretrainingDataset  # Assuming your dataset code is here
import os

# Hyperparameters (defined in config or here)
VOCAB_SIZE = 800  # Adjust based on your tokenizer vocabulary size
D_MODEL = 512  # Model dimension
SEQ_LEN = 512  # Maximum sequence length
N_HEADS = 8  # Number of attention heads
HIDDEN_DIM = 2048  # Hidden dimension in feedforward layers
N_LAYERS = 6  # Number of decoder layers
BATCH_SIZE = 16  # Adjust according to GPU memory
EPOCHS = 10  # Number of training epochs
LEARNING_RATE = 5e-5  # Learning rate for AdamW
MODEL_SAVE_PATH = "checkpoints/"  # Path to save model checkpoints

# Ensure checkpoint directory exists
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Load Tokenizer
tokenizer = load_tokenizer("multilingual_tokenizer")  # Ensure your tokenizer is trained and saved

# Prepare Dataset and DataLoader
dataset = MultilingualPretrainingDataset(tokenized_files=['data/tokenized_english.pt', 'data/tokenized_french.pt'], seq_len=SEQ_LEN)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Initialize Model
model = TransformerDecoder(seq_len=SEQ_LEN, d_model=D_MODEL, n_head=N_HEADS, hidden_dim=HIDDEN_DIM, dec_voc_size=VOCAB_SIZE, n_layers=N_LAYERS)
model.to('cuda')

# Loss Function (ignore padding index, typically 0)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Optimizer (AdamW is commonly used for transformer models)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Mixed Precision Training (optional but recommended for faster training)
scaler = GradScaler()

# Learning Rate Scheduler (optional)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(data_loader), epochs=EPOCHS)

### Training Loop ###
def train_model(model, data_loader, criterion, optimizer, scaler, scheduler, epochs=EPOCHS, save_path=MODEL_SAVE_PATH):
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = 0

        for step, (input_ids, target_ids) in enumerate(data_loader):
            input_ids = input_ids.cuda()
            target_ids = target_ids.cuda()

            optimizer.zero_grad()  # Reset gradients

            # Forward pass with mixed precision
            with autocast():
                outputs = model(input_ids)
                # Shift the outputs by 1 and flatten to match target_ids for loss calculation
                loss = criterion(outputs.view(-1, VOCAB_SIZE), target_ids.view(-1))

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(data_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(save_path, f"transformer_decoder_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

# Start Training
train_model(model, data_loader, criterion, optimizer, scaler, scheduler)
