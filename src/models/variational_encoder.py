import torch
from PIL import Image
from torchvision import transforms
import torch
import sys
import os
from torch.amp import autocast
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

current_dir = os.path.dirname(os.path.abspath(__file__))
latent_diffusion_path = os.path.join(current_dir, 'latent-diffusion')

# Add 'latent-diffusion' to sys.path if it's not already there
if latent_diffusion_path not in sys.path:
    sys.path.append(latent_diffusion_path)

class VAE():
    """
    Loads a VAE model from https://github.com/CompVis/taming-transformers

    Args:
        config_path (str): Path to the config file
        checkpoint_path (str): Path to the checkpoint file

    Functions:
        encode_decode_image(input_path, output_path, patch_size=256, overlap=0, debug=False): Encodes and decodes an image from a file
        encode_image(image, patch_size=0, overlap=0, debug=False): Encodes an image
        decode_latents(latents, patch_size=0, overlap=0, image_shape=None, debug=False): Decodes latents
    """


    def __init__(self, config_path="latent-diffusion/models/first_stage_models/vq-f8-n256/config.yaml", checkpoint_path="/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/model.ckpt", device='cuda'):
        self.config_path = config_path  # Chemin du fichier config
        self.config = OmegaConf.load(config_path)
        self.checkpoint_path = checkpoint_path  # Chemin des poids du modèle
        print(f"Loading model from {checkpoint_path}")
        self.model = instantiate_from_config(self.config.model)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def encode_decode_image(self, input_path, output_path, patch_size=256, overlap=0, debug=False):
        # Load and preprocess the image
        image = Image.open(input_path).convert("RGB")
        transform = transforms.ToTensor()
        input_tensor = transform(image).to('cuda')  # Shape: [C, H, W]

        C, H, W = input_tensor.shape

        # Calculate stride
        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError("Overlap must be smaller than the patch size.")

        # Determine patch positions
        y_steps = list(range(0, H, stride))
        x_steps = list(range(0, W, stride))

        # Adjust the last patch position to ensure full coverage
        if y_steps[-1] + patch_size > H:
            y_steps[-1] = H - patch_size
        if x_steps[-1] + patch_size > W:
            x_steps[-1] = W - patch_size

        # Remove potential duplicate positions caused by adjustment
        y_steps = sorted(list(set(y_steps)))
        x_steps = sorted(list(set(x_steps)))
        if debug:
            print(f"Image Dimensions: H={H}, W={W}")
            print(f"Patches will be extracted at Y positions: {y_steps}")
            print(f"Patches will be extracted at X positions: {x_steps}")

        patches = []
        positions = []
        for y in y_steps:
            for x in x_steps:
                patch = input_tensor[:, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((y, x))
        if debug:
            print(f"Total patches extracted: {len(patches)}")

        decoded_patches = []
        with torch.no_grad():
            for idx, patch in enumerate(patches):
                # Encode the patch
                latents = self.model.encode(patch.unsqueeze(0))[0]
                print(latents.shape)
                # Decode the latent representation
                decoded = self.model.decode(latents)
                # Ensure decoded patch has the exact patch_size
                decoded = decoded[:, :, :patch_size, :patch_size]
                decoded_patches.append(decoded)
                if debug:
                    if idx % 10 == 0:
                        print(f"Processed patch {idx+1}/{len(patches)}")

        # Reconstruct the image
        decoded_image = torch.zeros_like(input_tensor)
        overlap_count = torch.zeros_like(input_tensor)

        for patch, (y, x) in zip(decoded_patches, positions):
            # Get the actual height and width of the decoded patch
            _, _, patch_height, patch_width = patch.shape

            # Ensure the target slice does not exceed the image dimensions
            y_end = min(y + patch_height, decoded_image.shape[1])
            x_end = min(x + patch_width, decoded_image.shape[2])

            # Calculate the regions to assign
            assign_region = decoded_image[:, y:y_end, x:x_end]
            patch_region = patch.squeeze(0)[:, :y_end - y, :x_end - x]

            # Handle overlapping regions by averaging
            decoded_image[:, y:y_end, x:x_end] += patch_region
            overlap_count[:, y:y_end, x:x_end] += 1

        # Avoid division by zero
        overlap_count[overlap_count == 0] = 1

        # Average the overlapping regions
        decoded_image /= overlap_count

        # Post-process and save the decoded image
        decoded_image = decoded_image.cpu().clamp(0, 1)
        save_transform = transforms.ToPILImage()
        save_image = save_transform(decoded_image)
        save_image.save(output_path)
        if debug:
            print(f"Reconstructed image saved to {output_path}")
        return decoded_image
    
    def encode_image(self, image, patch_size=0, overlap=0, debug=False):
         # Load and preprocess the image
        transform = transforms.ToTensor()
        # input_tensor = transform(image).to('cuda')  # Shape: [C, H, W]
        input_tensor = image.to('cuda')

        if patch_size == 0:
            # Encode the full image
            with torch.no_grad():
                latents = self.model.encode(input_tensor)[0]
            return latents
        
        C, H, W = input_tensor.shape

        # Calculate stride
        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError("Overlap must be smaller than the patch size.")

        # Determine patch positions
        y_steps = list(range(0, H, stride))
        x_steps = list(range(0, W, stride))

        # Adjust the last patch position to ensure full coverage
        if y_steps[-1] + patch_size > H:
            y_steps[-1] = H - patch_size
        if x_steps[-1] + patch_size > W:
            x_steps[-1] = W - patch_size

        # Remove potential duplicate positions caused by adjustment
        y_steps = sorted(list(set(y_steps)))
        x_steps = sorted(list(set(x_steps)))
        if debug:
            print(f"Image Dimensions: H={H}, W={W}")
            print(f"Patches will be extracted at Y positions: {y_steps}")
            print(f"Patches will be extracted at X positions: {x_steps}")

        patches = []
        positions = []
        for y in y_steps:
            for x in x_steps:
                patch = input_tensor[:, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((y, x))
        if debug:
            print(f"Total patches extracted: {len(patches)}")

        encoded_patches = []
        with torch.no_grad():
            for idx, patch in enumerate(patches):
                # Encode the patch
                latents = self.model.encode(patch.unsqueeze(0))[0]
                encoded_patches.append(latents)
        
        return encoded_patches
    
    def decode_latents(self, latents, patch_size=0, overlap=0, image_shape=None, debug=False):
        if patch_size == 0:
            # Decode the full image
            with torch.no_grad():
                decoded = self.model.decode(latents)
            return decoded
        
        if image_shape is None:
            raise ValueError("Image shape must be provided when decoding patches.")
        H, W = image_shape

        # Calculate stride
        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError("Overlap must be smaller than the patch size.")

        # Determine patch positions
        y_steps = list(range(0, H, stride))
        x_steps = list(range(0, W, stride))

        # Adjust the last patch position to ensure full coverage
        if y_steps[-1] + patch_size > H:
            y_steps[-1] = H - patch_size
        if x_steps[-1] + patch_size > W:
            x_steps[-1] = W - patch_size

        # Remove potential duplicate positions caused by adjustment
        y_steps = sorted(list(set(y_steps)))
        x_steps = sorted(list(set(x_steps)))
        if debug:
            print(f"Image Dimensions: H={H}, W={W}")
            print(f"Patches will be extracted at Y positions: {y_steps}")
            print(f"Patches will be extracted at X positions: {x_steps}")

        positions = []
        for y in y_steps:
            for x in x_steps:
                positions.append((y, x))
        if debug:
            print(f"Total patches : {len(positions[0])}")

        decoded_patches = []
        with torch.no_grad():
            for idx, latent in enumerate(latents):
                # Decode the latent representation
                decoded = self.model.decode(latent)
                # Ensure decoded patch has the exact patch_size
                decoded = decoded[:, :, :patch_size, :patch_size]
                decoded_patches.append(decoded)
                if debug:
                    if idx % 10 == 0:
                        print(f"Processed patch {idx+1}/{len(latents)}")

        # Reconstruct the image
        decoded_image = torch.zeros((3, H, W)).to('cuda')
        overlap_count = torch.zeros((3, H, W)).to('cuda')

        for patch, (y, x) in zip(decoded_patches, positions):
            # Get the actual height and width of the decoded patch
            _, _, patch_height, patch_width = patch.shape

            # Ensure the target slice does not exceed the image dimensions
            y_end = min(y + patch_height, decoded_image.shape[1])
            x_end = min(x + patch_width, decoded_image.shape[2])

            # Calculate the regions to assign
            assign_region = decoded_image[:, y:y_end, x:x_end]
            patch_region = patch.squeeze(0)[:, :y_end - y, :x_end - x]

            # Handle overlapping regions by averaging
            decoded_image[:, y:y_end, x:x_end] += patch_region
            overlap_count[:, y:y_end, x:x_end] += 1

        # Avoid division by zero
        overlap_count[overlap_count == 0] = 1

        # Average the overlapping regions
        decoded_image /= overlap_count

        # Post-process and save the decoded image
        decoded_image = decoded_image.cpu().clamp(0, 1)
        return decoded_image
        
        
