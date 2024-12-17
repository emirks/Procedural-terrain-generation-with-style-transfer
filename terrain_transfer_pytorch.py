import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
from tqdm import tqdm
import os
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
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        print("Model structure:")
        print(vgg19)
        # Style and content layers exactly as specified in paper Section 2.2
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

def load_image(image_path, size=1500):  # Changed to paper's resolution
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
    
    image = loader(image).unsqueeze(0)
    image = image * 255  # Convert [0,1] to [0,255]
    mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(device)
    image = image.to(device)
    image = image[:, [2,1,0], :, :]  # RGB to BGR
    image = image - mean
    return image

def gram_matrix(tensor):
    """Calculate Gram Matrix as per paper Section 2.2, equation 2"""
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t)
    return gram

def compute_content_loss(gen_features, content_features):
    """Compute content loss as per paper Section 2.2, equation 1"""
    return F.mse_loss(gen_features, content_features)

def compute_style_loss(gen_features, style_features):
    """Compute style loss exactly as in transfer_morphology"""
    b, c, h, w = gen_features.size()
    gen_gram = gram_matrix(gen_features)
    style_gram = gram_matrix(style_features)
    
    # Normalize by all dimensions as in original paper
    return torch.sum((gen_gram - style_gram) ** 2) / (4 * (c * h * w) ** 2)

def total_variation_loss(image):
    """Calculate total variation loss as per paper Section 3, equation 5"""
    diff_i = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return diff_i + diff_j

def save_image(tensor, filename):
    image = tensor.clone().detach()
    image = image.squeeze(0)
    
    mean = torch.tensor([103.939, 116.779, 123.68]).view(3, 1, 1).to(device)
    image = image + mean
    image = image[[2,1,0], :, :]  # BGR to RGB
    image = image / 255.0
    
    image = image.cpu()
    image = torch.clamp(image, 0, 1)
    transforms.ToPILImage()(image).save(filename)

def main(args):
    try:
        os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
        
        if not os.path.exists(args.content_path):
            raise FileNotFoundError(f"Content image not found: {args.content_path}")
        if not os.path.exists(args.style_path):
            raise FileNotFoundError(f"Style image not found: {args.style_path}")
        
        content_img = load_image(args.content_path)
        style_img = load_image(args.style_path)
        
        gen_img = content_img.clone().requires_grad_(True)
        
        # Paper's exact optimizer settings (Section 4)
        initial_lr = 150.0
        optimizer = optim.SGD([gen_img], lr=initial_lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,  # decay_steps=100
            gamma=0.96      # decay_rate=0.96
        )
        
        # Paper's exact loss weights (Section 3, equation 7)
        content_weight = 2.5e-11  # α
        style_weight = 1e-5       # β
        tv_weight = 1e-10        # γ
        
        model = VGG19Features().to(device)
        
        content_features = model(content_img)
        style_features = model(style_img)
        
        best_loss = float('inf')
        best_img = None
        
        with tqdm(range(args.iterations)) as pbar:
            for i in pbar:
                optimizer.zero_grad()
                
                gen_features = model(gen_img)
                
                content_loss = compute_content_loss(
                    gen_features[model.content_layer],
                    content_features[model.content_layer]
                )
                
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
                
                total_loss = (content_weight * content_loss + 
                            style_weight * style_loss + 
                            tv_weight * tv_loss)
                
                if i % 100 == 0:
                    print(f"\nIteration {i}")
                    print(f"Content loss: {content_loss.item():.4f}")
                    print(f"Style loss: {style_loss.item():.4f}")
                    print(f"TV loss: {tv_loss.item():.4f}")
                    print(f"Total loss: {total_loss.item():.4f}")
                    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                    print(f"Generated image min/max: {gen_img.min().item():.2f}/{gen_img.max().item():.2f}")
                
                total_loss.backward()
                optimizer.step()
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_img = gen_img.clone()
                
                pbar.set_postfix({
                    'total_loss': f'{total_loss.item():.2f}',
                    'content': f'{(content_weight * content_loss).item():.2f}',
                    'style': f'{(style_weight * style_loss).item():.2f}',
                    'tv': f'{(tv_weight * tv_loss).item():.2f}'
                })
                
                if (i + 1) % 200 == 0:
                    save_image(gen_img, f"{args.output_prefix}_iter_{i+1}.png")
                
                if (i + 1) % 100 == 0:
                    scheduler.step()
        
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
    
    args = parser.parse_args()
    main(args)