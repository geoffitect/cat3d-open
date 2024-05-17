import collections
import jax.numpy as jnp

Rays = collections.namedtuple("Rays", ["origins", "directions"])

def compute_psnr(mse):
    """Compute PSNR given MSE."""
    return -10.0 * jnp.log(mse) / jnp.log(10.0)

def get_ray_directions(H, W, focal):
    """Get ray directions for a given image size and focal length."""
    i, j = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    dirs = jnp.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -jnp.ones_like(i)], axis=-1)
    return dirs

def get_rays(directions, c2w):
    """Get ray origins and directions from camera pose."""
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T
    rays_d = rays_d / jnp.linalg.norm(rays_d, axis=-1, keepdims=True)
    
    # The origin of all rays is the camera origin in world coordinate
    rays_o = jnp.broadcast_to(c2w[:3, 3], rays_d.shape)
    
    return rays_o, rays_d

def pose_spherical(theta, phi, radius):
    """Generate spherical rendering poses."""
    trans_t = lambda t: jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=jnp.float32)
    
    rot_phi = lambda phi: jnp.array([
        [1, 0, 0, 0],
        [0, jnp.cos(phi), -jnp.sin(phi), 0],
        [0, jnp.sin(phi), jnp.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=jnp.float32)
    
    rot_theta = lambda th: jnp.array([
        [jnp.cos(th), 0, -jnp.sin(th), 0],
        [0, 1, 0, 0],
        [jnp.sin(th), 0, jnp.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=jnp.float32)
    
    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    c2w = jnp.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

def load_blender(data_dir):
    """Load Blender dataset."""
    # TODO: Implement Blender dataset loading
    pass

def load_llff(data_dir):
    """Load LLFF dataset."""
    # TODO: Implement LLFF dataset loading
    pass

# Add any other necessary utility functions