import pandas as pd
import loguru
from loguru import logger
from data.dataset import Sentinel2Dataset
from data.loader import define_loaders
from utils.utils import load_config
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from model_zoo.models import define_model
import torch.optim as optim
from utils.torch import count_parameters, seed_everything
from training.losses import masked_mse_loss
# Configure loguru logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger.add(f"{log_dir}/training.log", rotation="10 MB")

config = load_config(config_path="cfg/config.yaml")
BASE_DIR = config["DATASET"]["base_dir"]
VERSION = config['DATASET']['version']
BATCH_SIZE = config['TRAINING']['batch_size']
VAL_BS = config["TRAINING"]['val_bs']
NUM_WORKERS = config['TRAINING']['num_workers']
RESIZE = config['TRAINING']['resize']
LEARNING_RATE = config['TRAINING']['learning_rate']
NUM_EPOCHS = config['TRAINING']['n_epoch']
SEED = config['TRAINING']['seed']
train_path = f"{BASE_DIR}/{VERSION}/train_path.csv"
val_path = f"{BASE_DIR}/{VERSION}/val_path.csv"
test_path = f"{BASE_DIR}/{VERSION}/test_path.csv"
seed_everything(seed=SEED)
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

train_dataset = Sentinel2Dataset(df_path=df_train,
                                 train=True, augmentation=False,
                                 img_size=RESIZE)

val_dataset = Sentinel2Dataset(df_path=df_val,
                               train=False, augmentation=False,
                               img_size=RESIZE)

train_loader, val_loader = define_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        val_bs=VAL_BS,
        num_workers=NUM_WORKERS,
    )



model = define_model(name="Unet", encoder_name="resnet34",
                     in_channel=3, out_channels=3, activation=None)

nb_parameters = count_parameters(model=model)
logger.info("Number of parameters: {}".format(nb_parameters))
print("Number of parameters: {}".format(nb_parameters))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# # Define loss function
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=float(LEARNING_RATE))

# Training parameters
best_epoch = 0
best_val_loss = float('inf')
step = 0
metrics_dict = {
    'train_loss': [],
    'val_loss': []
}
save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)
# Training loop
logger.info(f"Starting training for {NUM_EPOCHS} epochs")

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0

    with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE), colour='#3eedc4') as t:
        t.set_description('epoch: {}/{}'.format(epoch, NUM_EPOCHS - 1))

        for batch_idx, (x_data, y_data, valid_mask) in enumerate(train_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            valid_mask = valid_mask.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_data)

            # Calculate loss
            loss = masked_mse_loss(outputs, y_data, valid_mask)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            batch_loss = loss.item()
            train_loss += batch_loss

            # Update progress bar
            t.set_postfix(loss=batch_loss)
            t.update(x_data.size(0))

            step += 1

    avg_train_loss = train_loss / len(train_loader)
    metrics_dict['train_loss'].append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_dataset), colour='#f4d160') as t:
            t.set_description('validation')

            for batch_idx, (x_data, y_data, valid_mask) in enumerate(val_loader):
                x_data = x_data.to(device)
                y_data = y_data.to(device)
                valid_mask = valid_mask.to(device)

                # Forward pass
                outputs = model(x_data)

                # Calculate loss
                loss = masked_mse_loss(outputs, y_data, valid_mask)

                # Update statistics
                batch_loss = loss.item()
                val_loss += batch_loss

                # Update progress bar
                t.set_postfix(loss=batch_loss)
                t.update(x_data.size(0))

    avg_val_loss = val_loss / len(val_loader)
    metrics_dict['val_loss'].append(avg_val_loss)

    # Log epoch results
    logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        logger.info(f"Saved best model with Val Loss: {best_val_loss:.6f} at epoch {best_epoch+1}")

# Save final model
torch.save(model.state_dict(), f"{save_path}/final_model.pth")
logger.info(f"Training completed. Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch+1}")

# Save metrics
import matplotlib.pyplot as plt
import numpy as np

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(metrics_dict['train_loss'], label='Train Loss')
plt.plot(metrics_dict['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(f"{save_path}/loss_curves.png")
logger.info(f"Loss curves saved to {save_path}/loss_curves.png")

# Optionally, test the model on a few validation samples
model.eval()
with torch.no_grad():
    for i, (x_data, y_data) in enumerate(val_loader):
        if i >= 12:
            break

        x_data = x_data.to(device)
        y_data = y_data.to(device)

        output = model(x_data)

        # Convert to numpy for visualization
        x_np = x_data.cpu().numpy()[0].transpose(1, 2, 0)  # First image in batch, CHW to HWC
        y_np = y_data.cpu().numpy()[0].transpose(1, 2, 0)
        pred_np = output.cpu().numpy()[0].transpose(1, 2, 0)
        # Clip values to valid range for visualization
        x_np = np.clip(x_np, 0, 1)
        y_np = np.clip(y_np, 0, 1)
        pred_np = np.clip(pred_np, 0, 1)

        # Plot and save
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(x_np)
        axs[0].set_title('L1C Input')
        axs[1].imshow(pred_np)
        axs[1].set_title('Model Output')
        axs[2].imshow(y_np)
        axs[2].set_title('L2A Ground Truth')

        plt.savefig(f"{save_path}/sample_{i}_prediction.png")
        plt.close()

logger.info("Testing completed. Sample predictions saved.")
