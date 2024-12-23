import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from diffusers import LDMPipeline, UNet2DModel, VQModel, DDIMScheduler
import torch.nn.utils.prune as prune
from tqdm import tqdm
import torch.nn.functional as F

# ============================
# 1. Configuration Parameters
# ============================

# Paths
DATASET_PATH = "dataset/celeba_hq_small"
PRUNED_UNET_PATH = "pruned_unet_celebahq.pth"
PRUNED_VQVAE_PATH = "pruned_vqvae_celebahq.pth"
FINE_TUNED_UNET_PATH_TEMPLATE = "run/finetuned/fine_tuned_pruned_unet_epoch_{}.pth"
FINE_TUNED_VQVAE_PATH_TEMPLATE = "run/finetuned/fine_tuned_pruned_vqvae_epoch_{}.pth"
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pruning
PRUNE_AMOUNT = 0.3  # 30% pruning

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_WORKERS = 4
IMAGE_SIZE = 256  # Assuming CelebA-HQ images are 256x256

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 2. Define the Custom Dataset
# ============================

class CelebaHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, img) for img in os.listdir(root_dir)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image

# ============================
# 3. Prepare DataLoader
# ============================

# Define transformations (resize and normalize as needed)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Initialize the dataset and dataloader
dataset = CelebaHQDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# ============================
# 4. Load and Prune the Models
# ============================

# Load the pretrained models and scheduler
print("Loading pre-trained models and scheduler...")
unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

# Pruning function for the U-Net and VQModel
def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # print(f"Pruned layer: {name} - amount: {amount}")
    return model

# Apply pruning
print(f"Applying {PRUNE_AMOUNT*100}% unstructured pruning to U-Net and VQ-VAE...")
unet = prune_model(unet, amount=PRUNE_AMOUNT)
vqvae = prune_model(vqvae, amount=PRUNE_AMOUNT)

# Permanently remove pruning reparameterization to make pruning effective
def remove_pruning(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')
            # print(f"Removed pruning reparameterization for {module}")

remove_pruning(unet)
remove_pruning(vqvae)

# Save the pruned models
torch.save(unet.state_dict(), PRUNED_UNET_PATH)
torch.save(vqvae.state_dict(), PRUNED_VQVAE_PATH)
print(f"Pruned models saved as {PRUNED_UNET_PATH} and {PRUNED_VQVAE_PATH}")

# Calculate and print sparsity
def calculate_sparsity(model, model_name="Model"):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = 100.0 * zero_params / total_params
    print(f"Sparsity of {model_name}: {sparsity:.2f}% ({zero_params}/{total_params})")

calculate_sparsity(unet, "U-Net")
calculate_sparsity(vqvae, "VQ-VAE")

# ============================
# 5. Initialize the Pipeline
# ============================

# Initialize the pipeline with the pruned U-Net and VQ-VAE
print("Initializing the LDMPipeline with pruned models...")
pipeline = LDMPipeline(
    unet=unet,
    vqvae=vqvae,
    scheduler=scheduler,
).to(DEVICE)

# ============================
# 6. Define the Training Loop
# ============================

def kl_divergence_loss(noise_pred, noise):
    """
    Computes the KL Divergence between the predicted noise distribution and the actual noise.

    Parameters:
    - noise_pred (Tensor): Predicted noise by the model.
    - noise (Tensor): True Gaussian noise.

    Returns:
    - Tensor: KL Divergence loss value.
    """
    # Apply softmax to get probabilities (for demonstration, adjust if necessary)
    noise_pred_log_probs = F.log_softmax(noise_pred, dim=-1)
    noise_probs = F.softmax(noise, dim=-1)
    
    # Compute KL Divergence
    kl_loss = F.kl_div(noise_pred_log_probs, noise_probs, reduction='batchmean')
    return kl_loss

def generate_sample_images(pipeline, num_images=1, prompt="A high quality portrait", epoch=None):
    """
    Generate sample images and save them to the OUTPUT_DIR for manual viewing.

    Parameters:
    - pipeline (LDMPipeline): The diffusion model pipeline.
    - num_images (int): Number of images to generate.
    - prompt (str): The prompt for image generation.
    - epoch (int): Epoch number for saving images with epoch info.
    """
    pipeline.to(DEVICE)
    with torch.no_grad():
        generated_images = pipeline(batch_size=num_images, num_inference_steps=100).images
    
    # Save each generated image with epoch information in the filename
    for idx, img in enumerate(generated_images):
        filename = f"generated_epoch_{epoch}_image_{idx+1}.png" if epoch is not None else f"generated_image_{idx+1}.png"
        img.save(os.path.join(OUTPUT_DIR, filename))
        print(f"Image saved as {filename}")

# Move models to device
unet.to(DEVICE)
vqvae.to(DEVICE)

# Set models to training mode
unet.train()
vqvae.train()

# Define optimizer (only parameters that require gradients)
optimizer = optim.Adam(
    list(unet.parameters()) + list(vqvae.parameters()),
    lr=LEARNING_RATE
)

# Define a loss function, e.g., Mean Squared Error
criterion = nn.MSELoss()

# Training Loop
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        images = batch.to(DEVICE)
        
        # Forward pass through VQ-VAE to get latent representations
        with torch.no_grad():
            vqvae_output = vqvae.encode(images)
            # Corrected line: Use 'latent_sample' instead of 'latent_dist.sample()'
            latents = vqvae_output.latents
            latents = latents * vqvae.config.scaling_factor
        
        # Add noise according to the scheduler
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Forward pass through U-Net
        noise_pred = unet(noisy_latents, timesteps).sample
        
        # Compute loss
        loss = kl_divergence_loss(noise_pred, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Mask gradients of pruned weights
        with torch.no_grad():
            for module in unet.modules():
                if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                    zero_mask = module.weight == 0
                    if zero_mask.any():
                        module.weight.grad[zero_mask] = 0
            for module in vqvae.modules():
                if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                    zero_mask = module.weight == 0
                    if zero_mask.any():
                        module.weight.grad[zero_mask] = 0
        
        # Optimizer step
        optimizer.step()
        
        # Re-mask the weights to ensure pruned weights remain zero
        with torch.no_grad():
            for module in unet.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight[module.weight == 0] = 0
            for module in vqvae.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight[module.weight == 0] = 0
        
        # Accumulate loss
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_epoch_loss:.4f}")
    
    # Save the model at each epoch
    torch.save(unet.state_dict(), FINE_TUNED_UNET_PATH_TEMPLATE.format(epoch+1))
    torch.save(vqvae.state_dict(), FINE_TUNED_VQVAE_PATH_TEMPLATE.format(epoch+1))
    print(f"Saved fine-tuned models for epoch {epoch+1}")
    
    # Uncomment the following line to generate sample images after training
    generate_sample_images(pipeline, num_images=1, prompt="A high quality portrait", epoch=epoch+1)


print("\nTraining complete!")

# ============================
# 7. (Optional) Generate Sample Images
# ============================

# After training, you might want to generate some sample images to verify the fine-tuning

