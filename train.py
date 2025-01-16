import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from model.transformer_decoder import TransformerDecoder
from model.tokenizer.tokenizer_utils import load_tokenizer
from training.dataset import MultilingualPretrainingDataset
import os

# Hyperparameters 
VOCAB_SIZE = 800  # Adjust based on your tokenizer vocabulary size
D_MODEL = 512  # Model dimension
SEQ_LEN = 512  # Maximum sequence length
N_HEADS = 8  # Number of attention heads
HIDDEN_DIM = 2048  # Hidden dimension in feedforward layers
N_LAYERS = 6  # Number of decoder layers
BATCH_SIZE = 4  # Reduce batch size due to 4GB GPU memory constraint
EPOCHS = 1  # Number of training epochs
LEARNING_RATE = 5e-5  # Learning rate for AdamW
MODEL_SAVE_PATH = "checkpoints/"  # Path to save model checkpoints
ACCUMULATE_GRADIENTS = 4  # Gradient accumulation steps

# Ensure checkpoint directory exists
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

def main():
    # Load the tokenizer
    model_prefix = 'multilingual_tokenizer'
    tokenizer = load_tokenizer(model_prefix)

    # File containing raw text data
    input_file = './data/output_part1.txt'

    # List to store tokenized sentences
    tokenized_data = []

    # Define the maximum sequence length (optional, if padding/truncation is needed)
    max_seq_len = 100  # Adjust based on your model's input requirements

    # Read and tokenize the data
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            token_ids = tokenizer.encode(text, out_type=int)

            # Optional: Padding and truncation
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]  # Truncate
            else:
                token_ids += [tokenizer.pad_id()] * (max_seq_len - len(token_ids))  # Pad

            tokenized_data.append(token_ids)

    # Convert the tokenized data to a PyTorch tensor
    tokenized_tensor = torch.tensor(tokenized_data, dtype=torch.long)

    # Save the tensor to a file (tokenized_english.pt)
    os.makedirs('data', exist_ok=True)  # Create the 'data' directory if it doesn't exist
    torch.save(tokenized_tensor, 'data/tokenized_english.pt')

    # Prepare Dataset and DataLoader
    dataset = MultilingualPretrainingDataset(tokenized_files=['data/tokenized_english.pt'], seq_len=SEQ_LEN)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize Model
    model = TransformerDecoder(d_model=D_MODEL, n_head=N_HEADS, hidden_dim=HIDDEN_DIM, dec_voc_size=VOCAB_SIZE, n_layers=N_LAYERS).cuda()

    # Loss Function (ignore padding index, typically 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Optimizer (AdamW is commonly used for transformer models)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

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
                input_ids, target_ids = input_ids.cuda(), target_ids.cuda()  # Move to GPU

                optimizer.zero_grad()  # Reset gradients

                # Forward pass with mixed precision
                with autocast():
                    outputs = model(input_ids)
                    # Shift the outputs by 1 and flatten to match target_ids for loss calculation
                    loss = criterion(outputs.view(-1, VOCAB_SIZE), target_ids.view(-1))

                # Backward pass with mixed precision
                scaler.scale(loss).backward()

                # Gradient Accumulation
                if (step + 1) % ACCUMULATE_GRADIENTS == 0 or (step + 1) == len(data_loader):
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

# Protect the entry point for Windows
if __name__ == '__main__':
    main()
