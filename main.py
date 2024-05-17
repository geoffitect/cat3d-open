"""CAT3D: Create Anything in 3D with Multi-View Diffusion Models"""

import torch
from torch import nn
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import lpips
from . import utils
import os
import matplotlib as plt

# Constants 
NUM_DIFFUSION_STEPS = 1000
NUM_VIEWS = 8
IMAGE_SIZE = 512
LATENT_SIZE = 64
NUM_SAMPLES = 128
NEAR_BOUND = 0.5
FAR_BOUND = 1e5


# Datasets

class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class with ray generation method"""
    
    def get_rays(self, poses):
        """Generate rays for a batch of poses."""
        H, W = self.img_hw
        focal = self.focal
        
        dirs = utils.get_ray_directions(H, W, focal)
        rays_o, rays_d = utils.get_rays(dirs, poses)
        rays = utils.Rays(rays_o, rays_d)
        
        return rays

class ObjaverseDataset(BaseDataset):
    def __init__(self, data_dir, split, num_views=3):
        self.data_dir = data_dir
        self.split = split
        self.num_views = num_views
        
        # Load dataset metadata
        self.metadata = np.load(os.path.join(data_dir, f"{split}_metadata.npz"))
        self.object_ids = self.metadata["object_ids"]
        
    def __len__(self):
        return len(self.object_ids)
    
    def __getitem__(self, idx):
        object_id = self.object_ids[idx]
        
        # Load images and poses
        images = []
        poses = []
        for view in range(self.num_views):
            image_path = os.path.join(self.data_dir, f"{object_id}_{view}.png")
            image = plt.imread(image_path)
            images.append(image)
            
            pose_path = os.path.join(self.data_dir, f"{object_id}_{view}_pose.txt")
            pose = np.loadtxt(pose_path)
            poses.append(pose)
        
        images = np.stack(images, axis=0)
        poses = np.stack(poses, axis=0)
        
        return images, poses


class CO3DDataset(BaseDataset):
    def __init__(self, data_dir, split, num_views=3):
        self.data_dir = data_dir
        self.split = split
        self.num_views = num_views
        
        # Load dataset metadata
        self.metadata = np.load(os.path.join(data_dir, f"{split}_metadata.npz"))
        self.sequence_ids = self.metadata["sequence_ids"]
        
    def __len__(self):
        return len(self.sequence_ids)
    
    def __getitem__(self, idx):
        sequence_id = self.sequence_ids[idx]
        
        # Load images and poses
        images = []
        poses = []
        for view in range(self.num_views):
            image_path = os.path.join(self.data_dir, sequence_id, f"{view}.png")
            image = plt.imread(image_path)
            images.append(image)
            
            pose_path = os.path.join(self.data_dir, sequence_id, f"{view}_pose.txt")
            pose = np.loadtxt(pose_path)
            poses.append(pose)
        
        images = np.stack(images, axis=0)
        poses = np.stack(poses, axis=0)
        
        return images, poses


class RealEstate10KDataset(BaseDataset):
    def __init__(self, data_dir, split, num_views=3):
        self.data_dir = data_dir
        self.split = split
        self.num_views = num_views
        
        # Load dataset metadata
        self.metadata = np.load(os.path.join(data_dir, f"{split}_metadata.npz"))
        self.sequence_ids = self.metadata["sequence_ids"]
        
    def __len__(self):
        return len(self.sequence_ids)
    
    def __getitem__(self, idx):
        sequence_id = self.sequence_ids[idx]
        
        # Load images and poses
        images = []
        poses = []
        for view in range(self.num_views):
            image_path = os.path.join(self.data_dir, sequence_id, f"frame_{view}.png")
            image = plt.imread(image_path)
            images.append(image)
            
            pose_path = os.path.join(self.data_dir, sequence_id, f"frame_{view}_pose.txt")
            pose = np.loadtxt(pose_path)
            poses.append(pose)
        
        images = np.stack(images, axis=0)
        poses = np.stack(poses, axis=0)
        
        return images, poses


class MVImgNetDataset(BaseDataset):
    def __init__(self, data_dir, split, num_views=3):
        self.data_dir = data_dir
        self.split = split
        self.num_views = num_views
        
        # Load dataset metadata
        self.metadata = np.load(os.path.join(data_dir, f"{split}_metadata.npz"))
        self.sequence_ids = self.metadata["sequence_ids"]
        
    def __len__(self):
        return len(self.sequence_ids)
    
    def __getitem__(self, idx):
        sequence_id = self.sequence_ids[idx]
        
        # Load images and poses
        images = []
        poses = []
        for view in range(self.num_views):
            image_path = os.path.join(self.data_dir, sequence_id, f"view_{view}.png")
            image = plt.imread(image_path)
            images.append(image)
            
            pose_path = os.path.join(self.data_dir, sequence_id, f"view_{view}_pose.txt")
            pose = np.loadtxt(pose_path)
            poses.append(pose)
        
        images = np.stack(images, axis=0)
        poses = np.stack(poses, axis=0)
        
        return images, poses


class Encoder(nn.Module):
    """Encodes input images into latent space."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv(3, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv(64, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1) 
        self.conv4 = nn.Conv(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv5 = nn.Conv(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        
    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))  
        x = nn.relu(self.conv3(x))
        x = nn.relu(self.conv4(x))
        x = nn.relu(self.conv5(x))
        return x

class Decoder(nn.Module):
    """Decodes latents into output images."""  
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose(64, 64, kernel_size=(3, 3), padding=1) 
        self.deconv5 = nn.ConvTranspose(64, 3, kernel_size=(3, 3), padding=1)
        
    def __call__(self, z):
        z = nn.relu(self.deconv1(z))
        z = nn.relu(self.deconv2(z))
        z = nn.relu(self.deconv3(z))
        z = nn.relu(self.deconv4(z))
        z = self.deconv5(z)
        return z

class TemporalJointAttention(nn.Module):
    """Inflates 2D self-attention to 3D to connect multiple views."""
    def __init__(self, attn_module):
        super().__init__()
        self.attn_module = attn_module
        
    def __call__(self, x):
        B, T, _, H, W = x.shape
        x = x.reshape(B, T, -1, H*W) 
        x = x.permute(0, 2, 1, 3)  # (B, C, T, HW)
        x = self.attn_module(x)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1, H, W)
        return x
    
class MultiViewDiffusion(nn.Module):
    """Multi-view latent diffusion model."""
    def __init__(self):
        super().__init__()
        self.latent_dim = LATENT_SIZE*LATENT_SIZE*4
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Inflate 2D self-attention to 3D
        self.tempattn32 = TemporalJointAttention(nn.MultiHeadAttention(512, 4))
        self.tempattn16 = TemporalJointAttention(nn.MultiHeadAttention(512, 4))
        self.tempattn8 = TemporalJointAttention(nn.MultiHeadAttention(512, 4))
        
        self.input_proj = nn.Sequential(
            nn.Dense(4, 128),
            nn.relu,
            nn.Dense(128, 512)
        )
        
        self.diffusion_model = nn.Sequential(
            nn.Conv(512*2, 512, kernel_size=(3, 3), padding=1),
            nn.relu,
            nn.Conv(512, 512, kernel_size=(3, 3), padding=1),
            nn.relu,
            nn.Conv(512, 512, kernel_size=(3, 3), padding=1),
        )
        
    def __call__(self, x, t, camera_pose):
        """
        Forward pass of diffusion model.
        
        Args:
            x (Tensor): Input images, shape (B, T, C, H, W) 
            t (Tensor): Timesteps, shape (B,)
            camera_pose (Tensor): Camera poses, shape (B, T, 4)
        Returns:
            Tensor: Denoised latents, shape (B, T, D, H//8, W//8)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        z = self.encoder(x) 
        _, C_, H_, W_ = z.shape
        
        camera_pose = self.input_proj(camera_pose) 
        pose_embed = camera_pose.reshape(B, T, C_, 1, 1)
        pose_embed = pose_embed.expand(-1, -1, -1, H_, W_)
        z = z.reshape(B, T, C_, H_, W_)
        z = jnp.concatenate([z, pose_embed], axis=2)
        
        # Apply 3D attention 
        z = self.tempattn32(z)
        z = self.tempattn16(z)
        z = self.tempattn8(z)
        
        # Diffusion
        t_embed = self.get_timestep_embedding(t)
        t_embed = t_embed.reshape(B, 1, -1, 1, 1).expand(-1, T, -1, H_, W_)
        z = jnp.concatenate([z, t_embed], axis=2)
        z = z.reshape(B*T, -1, H_, W_)
        z = self.diffusion_model(z)
        z = z.reshape(B, T, -1, H_, W_)
        
        return z
    
    def get_loss(self, z_recon, z_true, t):
        loss = nn.mse_loss(z_recon, z_true)
        return loss
        
    def get_timestep_embedding(self, timesteps, dim=128):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps (Tensor): Timesteps to embed, shape (B,) 
            dim (int): Embedding dimension
        Returns:
            Tensor: Embeddings, shape (B, D)
        """
        assert len(timesteps.shape) == 1
        
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        
        emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        
        if dim % 2 == 1:  
            emb = jnp.pad(emb, [[0, 0], [0, 1]])
        
        return emb
    
class ZipNeRF(nn.Module):
    """NeRF model with Zip-NeRF architecture."""
    num_coarse_samples: int = 64
    num_fine_samples: int = 128
    use_viewdirs: bool = True
    near: float = NEAR_BOUND
    far: float = FAR_BOUND
    net_depth: int = 8
    net_width: int = 256
    activation: str = "relu"
    sigma_bias: float = -1.0
    
    @nn.compact
    def __call__(self, rng, rays, randomized, white_bkgd):
        
        def model_fn(rng, rays, num_samples, mlp, rgb_layer, sigma_layer):
            origins, directions = rays.origins, rays.directions
            
            z_vals = jnp.linspace(self.near, self.far, num_samples)
            
            if randomized:
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = jnp.concatenate([mids, z_vals[..., -1:]], axis=-1)
                lower = jnp.concatenate([z_vals[..., :1], mids], axis=-1)
                t_rand = random.uniform(rng, lower.shape)
                z_vals = lower + (upper - lower) * t_rand
                
            pts = origins[..., None, :] + z_vals[..., :, None] * directions[..., None, :]
            
            if self.use_viewdirs:
                viewdirs = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
                pts = jnp.concatenate([pts, viewdirs[..., None, :]], axis=-1)
                
            raw = mlp(pts)
            raw = jnp.reshape(raw, pts.shape[:-1] + (-1,))
            rgb = rgb_layer(raw)
            sigma = nn.relu(sigma_layer(raw))
            
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = jnp.concatenate([dists, jnp.broadcast_to([1e10], dists[..., :1].shape)], axis=-1)
            
            alpha = 1.0 - jnp.exp(-sigma * dists)
            trans = jnp.cumprod(1.0 - alpha + 1e-10, axis=-1, exclusive=True)
            weights = alpha * trans
            
            rgb_map = (weights[..., None] * rgb).sum(axis=-2)
            
            if white_bkgd:
                rgb_map += (1.0 - weights.sum(axis=-1, keepdims=True))
                
            depth_map = (weights * z_vals).sum(axis=-1)
            acc_map = weights.sum(axis=-1)
            
            return rgb_map, depth_map, acc_map
        
        coarse_mlp = nn.Sequential([nn.Dense(self.net_width) for _ in range(self.net_depth)],
                                   activation=self.activation)
        coarse_rgb_layer = nn.Dense(3)                             
        coarse_sigma_layer = nn.Dense(1, kernel_init=jax.nn.initializers.zeros,
                                      bias_init=lambda *_ : jnp.array([self.sigma_bias]))

        coarse_ret = model_fn(rng, rays, self.num_coarse_samples, 
                              coarse_mlp, coarse_rgb_layer, coarse_sigma_layer)
        
        if self.num_fine_samples <= 0:
            return (coarse_ret,)
        
        # Hierarchical sampling
        z_vals_mid = 0.5 * (coarse_ret[2][..., 1:] + coarse_ret[2][..., :-1])
        z_samples = sample_pdf(z_vals_mid, coarse_ret[2][..., 1:-1], self.num_fine_samples, det=randomized)
        z_samples = z_samples.detach()
        
        fine_rays = jax.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[1:]), rays)
        
        fine_mlp = nn.Sequential([nn.Dense(self.net_width) for _ in range(self.net_depth)],
                                 activation=self.activation)
        fine_rgb_layer = nn.Dense(3)
        fine_sigma_layer = nn.Dense(1, kernel_init=jax.nn.initializers.zeros,
                                    bias_init=lambda *_ : jnp.array([self.sigma_bias]))
        
        fine_ret = model_fn(rng, fine_rays._replace(origins=fine_rays.origins[..., None, :],
                                                    directions=fine_rays.directions[..., None, :]), 
                            self.num_fine_samples, fine_mlp, fine_rgb_layer, fine_sigma_layer)
        
        return coarse_ret, fine_ret


def generate_views(model, img, pose):
    """Generate novel views from a single input image."""
    rng = random.PRNGKey(0)
    
    with torch.no_grad():
        B = pose.shape[0]
        T = pose.shape[1] 
        
        z = model.encoder(img)
        z_in = z.repeat(1,T,1,1,1)
        
        z_out = jnp.zeros_like(z_in)
        for t in range(NUM_DIFFUSION_STEPS):
            t_batch = jnp.full((B,), t)
            _,z_out = model.apply({"params": model.params}, z_out, t_batch, pose, rngs={"dropout": rng})
            
        x_out = model.decoder(z_out.reshape(B*T,-1,LATENT_SIZE,LATENT_SIZE))
        x_out = x_out.reshape(B,T,3,IMAGE_SIZE,IMAGE_SIZE)
        
    return np.array(x_out)

def sample_pdf(bins, weights, N_samples, det=False):
   """Sample from 1D PDFs via the inversion method."""
   # Get pdf
   weights = weights + 1e-5 # prevent nans
   pdf = weights / jnp.sum(weights, axis=-1, keepdims=True)
   cdf = jnp.cumsum(pdf, axis=-1)
   cdf = jnp.concatenate([jnp.zeros_like(cdf[..., :1]), cdf], axis=-1)

   # Take uniform samples
   if det:
       u = jnp.linspace(0., 1., N_samples)
       u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
   else:
       u = random.uniform(random.PRNGKey(0), list(cdf.shape[:-1]) + [N_samples])

   # Invert CDF
   inds = jnp.searchsorted(cdf, u, side='right')
   below = jnp.maximum(0, inds-1)
   above = jnp.minimum(cdf.shape[-1]-1, inds)
   inds_g = jnp.stack([below, above], axis=-1)
   cdf_g = jnp.take_along_axis(cdf, inds_g, axis=-1)
   bins_g = jnp.take_along_axis(bins, inds_g, axis=-1)

   denom = (cdf_g[..., 1]-cdf_g[..., 0])
   denom = jnp.where(denom < 1e-5, jnp.ones_like(denom), denom)
   t = (u-cdf_g[..., 0])/denom
   samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
   return samples


def distortion_loss(z_vals, weights, det_jac):
   """Distortion loss to regularize geometry."""
   loss = (weights * det_jac[:,None]).sum(axis=1).mean()
   return 0.01 * loss

def normalized_l2_weight_decay(model):
   """Normalized L2 weight decay on NeRF parameters."""
   decay = 0
   for _, v in model.params.items():
       decay += (0.1 * jnp.sum(v**2) / np.prod(v.shape))        
   return decay

def robust_loss(x_recon, x_true, eps=1e-2):
   """Robust Charbonnier loss."""
   return jnp.sqrt((x_recon - x_true)**2 + eps**2)

def perceptual_loss(img1, img2):
    """Perceptual loss using LPIPS."""
    # Resize images to 256x256 if needed
    if img1.shape[-2:] != (256, 256):
        img1 = jax.image.resize(img1, (256, 256), method='bilinear')
    if img2.shape[-2:] != (256, 256):  
        img2 = jax.image.resize(img2, (256, 256), method='bilinear')
    
    # Normalize images to [-1, 1]  
    img1 = (img1 - 0.5) * 2
    img2 = (img2 - 0.5) * 2
    
    # Initialize LPIPS model if needed
    if not hasattr(perceptual_loss, 'lpips_model'):
        perceptual_loss.lpips_model = lpips.LPIPS(net='alex')
    
    # Compute perceptual distance
    dist = perceptual_loss.lpips_model(img1, img2)
    return dist.mean()

def sample_novel_poses(pose, n_novel):
    """Sample novel view poses based on the input poses."""
    # Assuming pose has shape (B, T, 4, 4) representing transformation matrices
    
    # Sample random rotations and translations
    rng = random.PRNGKey(0)
    rotations = random.normal(rng, (n_novel, 3, 3)) 
    translations = random.normal(rng, (n_novel, 3))
    
    # Compose new transformation matrices
    novel_poses = jnp.eye(4)[None, ...].repeat(n_novel, axis=0)
    novel_poses = novel_poses.at[:, :3, :3].set(rotations)
    novel_poses = novel_poses.at[:, :3, 3].set(translations)
    
    return novel_poses

def reconstruct_3d(model, diffusion_model, dataloader, optimizer, epochs):
    """Run 3D reconstruction with diffusion prior."""
    rng = random.PRNGKey(0)
    state = train_state.TrainState.create(apply_fn=model.apply, params=model.params, tx=optimizer)

    for epoch in range(epochs):
        for batch in dataloader:
            img, pose = batch
           
            # Sample novel views from diffusion model
            n_novel = 3 
            img_novel = generate_views(diffusion_model, img[:n_novel], pose[:n_novel])

            # Sample novel view poses
            pose_novel = sample_novel_poses(pose, n_novel)
            rays_novel = dataloader.get_rays(pose_novel)
           
            rays = dataloader.get_rays(pose)
           
            def loss_fn(params):
                ret = model.apply({"params": params}, rng, rays.origins, rays.directions) 
                
                rgb_coarse, _, _ = ret[0]
                loss_coarse = ((rgb_coarse - img)**2).mean()
                psnr_coarse = utils.compute_psnr(loss_coarse)
                
                if len(ret) > 1:
                    rgb_fine, _, _ = ret[1]
                    loss_fine = ((rgb_fine - img)**2).mean()
                    psnr_fine = utils.compute_psnr(loss_fine)
                    loss = loss_coarse + loss_fine
                else:
                    loss = loss_coarse
                    psnr_fine = 0
               
                charbonnier = lambda x: jnp.sqrt(x**2 + 1e-3**2) 
               
                rgb_novel_coarse, _, _ = model.apply({"params": params}, rng, 
                                                    rays_novel.origins, rays_novel.directions)[0]
                loss_novel = charbonnier(rgb_novel_coarse - img_novel).mean()
                loss_lpips = perceptual_loss(rgb_novel_coarse, img_novel)  
                loss += loss_novel + 0.25*loss_lpips
               
                decay = normalized_l2_weight_decay(model)
                loss += decay
               
                return loss, (psnr_fine, loss_coarse, psnr_coarse, decay)
           
            # Update NeRF  
            (loss, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params) # originally loss was _
            state = state.apply_gradients(grads=grad)
           
            psnr, loss_c, psnr_c, decay = metrics
            print(f"Epoch {epoch}, Loss: {loss:.4f}, PSNR: {psnr:.2f}, "
                 f"Loss_c: {loss_c:.4f}, PSNR_c: {psnr_c:.2f}, Weight Decay: {decay:.2e}")
           
        print(f"Epoch {epoch} completed.")
        model.params = state.params

def main():
    """Main function."""    
    model = ZipNeRF()
    
    diffusion_model = MultiViewDiffusion()
    model = model.to('mps') # For Apple Silicon GPU  
    diffusion_model_state = checkpoints.restore_checkpoint("diffusion_model_dir", None)
    diffusion_model = diffusion_model.load_state_dict(diffusion_model_state)
   
    # Prepare datasets
    train_datasets = {
        'Objaverse': ObjaverseDataset(...),
        'CO3D': CO3DDataset(...),
        'RealEstate10K': RealEstate10KDataset(...),
        'MVImgNet': MVImgNetDataset(...)
    }
    
    # Combine datasets 
    train_dataset = torch.utils.data.ConcatDataset(list(train_datasets.values()))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   
    reconstruct_3d(model, diffusion_model, train_dataloader, optimizer, epochs=2000)

if __name__ == '__main__':
   main()
