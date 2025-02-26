import torch
import sys
from torch.amp import autocast


custom_folder_path = "latent-diffusion"
sys.path.append(custom_folder_path)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# Charger le fichier de configuration
config_path = "latent-diffusion/models/first_stage_models/vq-f8-n256/config.yaml"  # Chemin potentiel du fichier config
config = OmegaConf.load(config_path)

# Instancier le modèle à partir de la configuration
def load_model_from_config(config, checkpoint_path):
    print(f"Loading model from {checkpoint_path}")
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model

# Chemin vers les poids du modèle pré-entraîné
checkpoint_path = "model.ckpt" 
model = load_model_from_config(config, checkpoint_path)
model = model.to('cuda')


from PIL import Image
import numpy as np
from torchvision import transforms

# Charger et pré-traiter une image
image = Image.open("benzema.jpg_large")
if image.mode != "RGB":
    image = image.convert("RGB")


preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
image = preprocess(image).unsqueeze(0).to('cuda')

import numpy as np
import torch.nn.functional as F

def split_into_patches(image, patch_size, stride):
    """
    Split an image into overlapping patches, including the last one, padding if necessary.
    """
    patches = []
    indices = []
    h, w = image.shape[-2:]

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            # Compute the end coordinates of the patch
            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)
            
            # Extract the patch
            patch = image[:,:, i:end_i, j:end_j]

            # If the patch is smaller than the patch_size, pad it
            if patch.shape[-2] < patch_size or patch.shape[-1] < patch_size:
                # Calculate padding
                pad_bottom = patch_size - patch.shape[-2]
                pad_right = patch_size - patch.shape[-1]
                padding = (0, pad_right, 0, pad_bottom)  # (left, right, top, bottom)
                patch = F.pad(patch, padding, mode='constant', value=0)
            patches.append(patch)
            indices.append((i, j))

    # Ensure that all patches have the same size
    # patches_tensor = torch.stack(patches, dim=0)  # Stack along the batch dimension

    return patches, indices


def reconstruct_from_patches(patches, indices, image_shape, patch_size, stride):
    """
    Reconstruct the full image from processed patches.
    """
    h, w = image_shape[-2:]
    reconstructed_image = np.zeros((3, h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for patch, (i, j) in zip(patches, indices):
        patch = patch.cpu().numpy()
        # Squeeze uniquement la dimension de batch si elle est de taille 1
        if patch.shape[0] == 1:
            patch = patch.squeeze(0)

        # Déterminer la taille valide du patch
        patch_height = patch.shape[1]
        patch_width = patch.shape[2]

        # afficher min et max de chaque channel pour tout le patch

        remaining_height = h - i
        remaining_width = w - j

        valid_height = min(patch_height, remaining_height)
        valid_width = min(patch_width, remaining_width)

        # Accumuler seulement la portion valide du patch
        reconstructed_image[:, i:i + valid_height, j:j + valid_width] += patch[:, :valid_height, :valid_width]
        weight_map[i:i + valid_height, j:j + valid_width] += 1

    # Éviter la division par zéro
    weight_map[weight_map == 0] = 1
    reconstructed_image /= weight_map
    # Remplacer les valeurs NaN si nécessaire
    reconstructed_image = np.nan_to_num(reconstructed_image, nan=0.0)

    return reconstructed_image

image = image.to('cuda')

# z = model.encoder(image)
# reconstructed_image = model.decoder(z)
# reconstructed_image = reconstructed_image.detach().cpu().squeeze(0).numpy()


patch_size = 512
stride = 256  # Overlap of 50%
patches, indices = split_into_patches(image, patch_size, stride)

model.eval()

# Encode patches in batches
encoded_patches = []
encoder = model.encoder
decoder = model.decoder

batch_size = 4  # Process patches in batches
with torch.no_grad():
    for i in range(0, len(patches), batch_size):
        batch = torch.cat(patches[i:i + batch_size]).to('cuda')
        with autocast("cuda", enabled=True):
            z = model.encode(batch)[0]
        encoded_patches.extend(z.cpu())  # Move to CPU to save GPU memory

# Free memory and move the decoder to GPU
model.encoder.to('cpu')
model.decoder.to('cuda')

# Decode patches in batches
decoded_patches = []
with torch.no_grad():
    for i in range(0, len(encoded_patches), batch_size):
        batch = torch.stack(encoded_patches[i:i + batch_size]).to('cuda')
        with autocast("cuda", enabled=True):
            reconstructed_batch = model.decode(batch)
        decoded_patches.extend(reconstructed_batch.cpu())  # Move to CPU

# Reconstruct the full image from patches
reconstructed_image = reconstruct_from_patches(decoded_patches, indices, image.shape, patch_size, stride)


# Save the reconstructed image

print("Avant dénormalisation:", reconstructed_image.min(), reconstructed_image.max())
# reconstructed_image = (reconstructed_image + 1) / 2.0  # Revenir à [0, 1]
print("Après dénormalisation:", reconstructed_image.min(), reconstructed_image.max())
# reconstructed_image = 1 - reconstructed_image  # Inverser les couleurs
# reconstructed_image = (reconstructed_image * 255).clip(0, 255).astype(np.uint8)  # [0, 255]
reconstructed_image = reconstructed_image.transpose(1, 2, 0)  # HWC format pour PIL
reconstructed_image = reconstructed_image[..., [0, 1, 2]]  # RGB

reconstructed_image = Image.fromarray(reconstructed_image, mode="RGB")

reconstructed_image.save("worc.jpg")

image = image.detach().squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
image = np.nan_to_num(image, nan=0.0)

# Denormalize the image
image = (image * 255).clip(0, 255).astype(np.uint8)

image = image.transpose(1, 2, 0)  # Convert to HWC format
image = image[..., [0, 1, 2]]  # Swap BGR to RGB

image = Image.fromarray(image, mode="RGB")
image.save("image-i.jpg")
