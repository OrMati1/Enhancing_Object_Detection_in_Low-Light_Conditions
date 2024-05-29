import os
import re
import datetime
from datetime import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import logging
from UNetv2_model import UNetv2

# Setup logging
current_time = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"training_log_{current_time}.txt"
skipped_log_filename = "skipped_images.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename, mode='w'),  # Log file with timestamp
                        logging.StreamHandler()
                    ])


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)  # Ensure the directory exists
        return None  # No checkpoint to return if the directory had to be created

    pattern = re.compile(r"unetv2_epoch_(\d+)\.pth")
    max_epoch = 0
    latest_checkpoint = None

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint = filename

    return os.path.join(checkpoint_dir, latest_checkpoint) if latest_checkpoint else None


def load_skipped_images(skipped_log_filename):
    if os.path.exists(skipped_log_filename):
        with open(skipped_log_filename, 'r') as f:
            skipped_images = f.read().splitlines()
        # Remove the header line "Skipped images:" if it exists
        if skipped_images and skipped_images[0] == "Skipped images:":
            skipped_images = skipped_images[1:]
    else:
        skipped_images = []
    return skipped_images


class LoLDataset(Dataset):
    def __init__(self, base_path, transform=None, subset_size=None, skipped_images=[]):
        self.high_dir = os.path.join(base_path, 'high')
        self.low_dir = os.path.join(base_path, 'low')
        self.transform = transform
        self.images = [f for f in os.listdir(self.low_dir) if f.endswith('.png') and f not in skipped_images]
        if subset_size:
            self.images = self.images[:subset_size]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        low_path = os.path.join(self.low_dir, img_name)
        high_path = os.path.join(self.high_dir, img_name)

        low_image = Image.open(low_path).convert('RGB')
        high_image = Image.open(high_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        return low_image, high_image


class ExDarkDataset(Dataset):
    def __init__(self, bright_light_path, low_light_path, transform=None, subset_size=None, skipped_images=[]):
        self.bright_light_path = os.path.normpath(bright_light_path)
        self.low_light_path = os.path.normpath(low_light_path)
        self.transform = transform
        self.images = sorted([f for f in os.listdir(self.low_light_path) if f.endswith(('.png', '.jpg', '.jpeg')) and f not in skipped_images])

        # Filter images within the specified range
        self.images = [f for f in self.images if self._in_range(f)]

        if subset_size:
            self.images = self.images[:subset_size]

    def _in_range(self, filename):
        match = re.match(r"2015_(\d{5})", filename)
        if match:
            num = int(match.group(1))
            return 1 <= num <= 1736
        return False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        low_path = os.path.join(self.low_light_path, img_name)
        bright_path = os.path.join(self.bright_light_path, img_name)

        try:
            low_image = Image.open(low_path).convert('RGB')
        except FileNotFoundError as e:
            if not os.path.exists(skipped_log_filename) or os.stat(skipped_log_filename).st_size == 0:
                logging.error(f"Low light image not found: {low_path}")
                with open(skipped_log_filename, 'a') as f:
                    f.write(f"{img_name}\n")
            raise e

        try:
            bright_image = Image.open(bright_path).convert('RGB')
        except FileNotFoundError:
            if not os.path.exists(skipped_log_filename) or os.stat(skipped_log_filename).st_size == 0:
                logging.warning(f"Bright light image not found: {bright_path}. Skipping this pair.")
                with open(skipped_log_filename, 'a') as f:
                    f.write(f"{img_name}\n")
            return self.__getitem__((idx + 1) % len(self.images))  # Continue to the next image

        if self.transform:
            low_image = self.transform(low_image)
            bright_image = self.transform(bright_image)

        return low_image, bright_image


def calculate_metrics(output, target):
    output_np = output.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
    psnr_values = []
    ssim_values = []

    for o, t in zip(output_np, target_np):
        psnr_val = psnr(t, o, data_range=t.max() - t.min())
        psnr_values.append(psnr_val)

        try:
            if min(o.shape[0], o.shape[1]) >= 7:
                win_size = min(7, o.shape[0], o.shape[1])
                if win_size % 2 == 0:
                    win_size -= 1
                ssim_val = ssim(t, o, data_range=t.max() - t.min(), channel_axis=2, win_size=win_size)
                ssim_values.append(ssim_val)
            else:
                ssim_values.append(np.nan)
        except Exception as e:
            logging.error(f"SSIM calculation failed: {e}")
            ssim_values.append(np.nan)

    avg_psnr = np.nanmean(psnr_values)
    avg_ssim = np.nanmean(ssim_values) if len(ssim_values) > 0 else np.nan

    return avg_psnr, avg_ssim


def get_dataloader(dataset_name, transform, subset_size, skipped_images):
    if dataset_name == 'LoL':
        dataset = LoLDataset(
            base_path='/database/database_lol/our485',
            transform=transform,
            subset_size=subset_size,
            skipped_images=skipped_images
        )
    elif dataset_name == 'ExDark':
        dataset = ExDarkDataset(
            bright_light_path='/training_data/bright_light/lowlight_enhancement',
            low_light_path='/training_data/low_light',
            transform=transform,
            subset_size=subset_size,
            skipped_images=skipped_images
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=1, shuffle=True)


def main(num_epochs=50, subset_size=None, dataset_name='LoL'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    # Load skipped images from the log file
    skipped_images = load_skipped_images(skipped_log_filename)

    dataloader = get_dataloader(dataset_name, transform, subset_size, skipped_images)

    model = UNetv2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    checkpoint_dir = '/model'
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint.get('model_state_dict', model.state_dict()))
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
        start_epoch = checkpoint.get('epoch', 0)
        logging.info(f"Resumed from checkpoint: {checkpoint_path}, starting at epoch {start_epoch}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss, total_psnr, total_ssim, count = 0.0, 0.0, 0.0, 0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            psnr_val, ssim_val = calculate_metrics(outputs, targets)
            if not np.isnan(ssim_val):
                total_ssim += ssim_val
                count += 1
            total_psnr += psnr_val

        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        avg_ssim = total_ssim / count if count > 0 else 0
        logging.info(f'Epoch {epoch + 1}, Loss: {avg_loss}, Avg PSNR: {avg_psnr}, Avg SSIM: {avg_ssim}')

        # Save checkpoint at the end of each epoch
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'unetv2_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        logging.info(f"Checkpoint saved at '{checkpoint_path}'")

    logging.info('Finished Training')


if __name__ == '__main__':
    main(num_epochs=100, subset_size=None, dataset_name='LoL')
