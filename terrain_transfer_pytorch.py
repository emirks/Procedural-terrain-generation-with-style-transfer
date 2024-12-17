import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import torch.nn.functional as F

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained VGG19 model with ImageNet weights
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        # Style and content layers matching paper specifications (Section 2.2)
        self.style_layers = {
            '0',   # conv1_1
            '5',   # conv2_1
            '10',  # conv3_1
            '19',  # conv4_1
            '28'   # conv5_1
        }
        self.content_layer = '30'  # conv5_2
        
        self.features = vgg19
        
        # Freeze the network
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.style_layers or name == self.content_layer:
                features[name] = x
        return features

def get_image_size(image_path, target_height=512):
    width, height = Image.open(image_path).size
    new_width = int(width * target_height / height)
    return target_height, new_width

def load_image(image_path, size=256):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize keeping aspect ratio
    aspect_ratio = image.size[0] / image.size[1]
    new_width = int(size * aspect_ratio)
    
    loader = transforms.Compose([
        transforms.Resize((size, new_width)),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess
    image = loader(image).unsqueeze(0)
    # VGG preprocessing
    image = image * 255  # Convert [0,1] to [0,255]
    mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(device)
    image = image.to(device)  # Move image to device before operations
    image = image[:, [2,1,0], :, :]  # RGB to BGR
    image = image - mean  # Subtract mean
    return image

def gram_matrix(tensor):
    """
    Calculate Gram Matrix with proper scaling
    """
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t)
    return gram.div(c * h * w)

def compute_content_loss(gen_features, content_features):
    """
    Compute content loss as per paper Section 2.2, equation 1
    """
    return torch.mean((gen_features - content_features) ** 2)

def compute_style_loss(gen_features, style_features):
    """
    Compute style loss with proper scaling
    """
    gen_gram = gram_matrix(gen_features)
    style_gram = gram_matrix(style_features)
    
    # Normalize by feature dimensions
    b, c, h, w = gen_features.size()
    return torch.mean((gen_gram - style_gram) ** 2)

def total_variation_loss(image):
    """
    Calculate total variation loss with proper normalization
    """
    b, c, h, w = image.size()
    diff_i = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return (diff_i + diff_j) / (b * c * h * w)

def save_image(tensor, filename):
    image = tensor.clone().detach()
    image = image.squeeze(0)
    
    # Undo VGG preprocessing
    mean = torch.tensor([103.939, 116.779, 123.68]).view(3, 1, 1).to(device)
    image = image + mean
    image = image[[2,1,0], :, :]  # BGR to RGB
    image = image / 255.0  # Scale to [0,1]
    
    # Move to CPU for saving
    image = image.cpu()
    
    # Clip values
    image = torch.clamp(image, 0, 1)
    
    # Save
    transforms.ToPILImage()(image).save(filename)


def main(args):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
        
        # Check if input files exist
        if not os.path.exists(args.content_path):
            raise FileNotFoundError(f"Content image not found: {args.content_path}")
        if not os.path.exists(args.style_path):
            raise FileNotFoundError(f"Style image not found: {args.style_path}")
        
        # Load images
        content_img = load_image(args.content_path)
        style_img = load_image(args.style_path)
        
        # Initialize generated image with content image
        gen_img = content_img.clone().requires_grad_(True)
        
        # Match paper's optimizer settings
        initial_lr = 150.0  # Much higher initial learning rate
        optimizer = optim.SGD([gen_img], lr=initial_lr)  # Using SGD instead of Adam
        
        # Create equivalent learning rate schedule
        # decay_rate = 0.96 every 100 steps
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,  # decay_steps=100
            gamma=0.96      # decay_rate=0.96
        )
        
        # Keep the same loss weights as they work well
        content_weight = 2.5e-5
        style_weight = 1e2
        tv_weight = 1e-5
      
        # # Add gradient clipping to prevent instability
        # torch.nn.utils.clip_grad_norm_([gen_img], max_norm=1.0)
        
        # Initialize model and optimizer
        model = VGG19Features().to(device)
        
        # Extract features once
        content_features = model(content_img)
        style_features = model(style_img)
        
        best_loss = float('inf')
        best_img = None
        
        # Training loop
        with tqdm(range(args.iterations)) as pbar:
            for i in pbar:
                optimizer.zero_grad()
                
                # Get features of generated image
                gen_features = model(gen_img)
                
                # Compute losses
                content_loss = compute_content_loss(
                    gen_features[model.content_layer],
                    content_features[model.content_layer]
                )
                
                # Compute style loss for each layer
                style_loss = 0
                for layer in model.style_layers:
                    layer_style_loss = compute_style_loss(
                        gen_features[layer],
                        style_features[layer]
                    )
                    style_loss += layer_style_loss
                    if i % 100 == 0:
                        print(f"Style loss for layer {layer}: {layer_style_loss.item():.4f}")
                
                style_loss /= len(model.style_layers)
                tv_loss = total_variation_loss(gen_img)
                
                # Total loss
                total_loss = (content_weight * content_loss + 
                            style_weight * style_loss + 
                            tv_weight * tv_loss)
                
                # Log detailed loss components
                if i % 100 == 0:
                    print(f"\nIteration {i}")
                    print(f"Content loss: {content_loss.item():.4f}")
                    print(f"Style loss: {style_loss.item():.4f}")
                    print(f"TV loss: {tv_loss.item():.4f}")
                    print(f"Total loss: {total_loss.item():.4f}")
                    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                    
                    # Log image statistics
                    print(f"Generated image min/max: {gen_img.min().item():.2f}/{gen_img.max().item():.2f}")
                
                total_loss.backward()
                optimizer.step()
                
                # Save best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_img = gen_img.clone()
                
                # Update progress bar with normalized values
                pbar.set_postfix({
                    'total_loss': f'{total_loss.item():.2f}',
                    'content': f'{(content_weight * content_loss).item():.2f}',
                    'style': f'{(style_weight * style_loss).item():.2f}',
                    'tv': f'{(tv_weight * tv_loss).item():.2f}'
                })
                
                # Save intermediate result
                if (i + 1) % 50 == 0:
                    save_image(gen_img, f"{args.output_prefix}_iter_{i+1}.png")
                
                if (i + 1) % 100 == 0:
                    scheduler.step()
        
        # Save the best result at the end
        if best_img is not None:
            save_image(best_img, f"{args.output_prefix}_best.png")
                    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terrain Style Transfer")
    parser.add_argument("--content_path", type=str, default="noise/perlin3.png",
                      help="Path to content image (noise map)")
    parser.add_argument("--style_path", type=str, default="sources/himalaya.jpg",
                      help="Path to style image (terrain height map)")
    parser.add_argument("--output_prefix", type=str, default="outputs/transferred",
                      help="Prefix for output files")
    parser.add_argument("--iterations", type=int, default=2000,
                      help="Number of iterations")
    parser.add_argument("--lr", type=float, default=0.1,
                      help="Learning rate")
    
    args = parser.parse_args()
    main(args)