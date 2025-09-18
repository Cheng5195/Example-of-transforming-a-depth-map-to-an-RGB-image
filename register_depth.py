import os
import cv2
import numpy as np
import transforms3d.quaternions as quat

def apply_distortion(x_norm, y_norm, D):
    """
    Apply distortion to normalized coordinates
    D: [k1, k2, p1, p2, k3, k4, k5, k6] 
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = D
    
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2
    r6 = r2**3
    
    # Radial distortion
    radial_factor = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
    
    # Tangential distortion
    x_distorted = x_norm * radial_factor + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
    y_distorted = y_norm * radial_factor + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
    
    return x_distorted, y_distorted

# ==== Paths ====
base_dir = os.path.join(os.path.dirname(__file__), "data")
rgb_dir = os.path.join(base_dir, "rgb")
depth_dir = os.path.join(base_dir, "depth")
assoc_path = os.path.join(base_dir, "associations_depth.txt")
output_dir = os.path.join(base_dir, "depth_registered4")
os.makedirs(output_dir, exist_ok=True)

# ==== Depth Camera Intrinsics ====
K_d = np.array([
    [252.1798095703125, 0.0, 258.2330017089844],
    [0.0, 252.27508544921875, 256.9632263183594],
    [0.0, 0.0, 1.0]
])
D_d = np.array([0.48829081654548645, -0.018839804455637932, 8.734838775126263e-05,
                -1.3474861589202192e-05, -0.0018967381911352277, 0.8293085098266602,
                0.0709189623594284, -0.010790164582431316])

# ==== RGB Camera Intrinsics (resized to 1024x768) ====
K_rgb_full = np.array([
    [968.3805541992188, 0.0, 1025.4171142578125],
    [0.0, 968.3682861328125, 780.155029296875],
    [0.0, 0.0, 1.0]
])
D_rgb = np.array([0.5562127828598022, -2.973095417022705, 0.0003248305292800069, 
                  -0.0003230230649933219, 1.753787636756897, 0.42559221386909485, 
                  -2.7683727741241455, 1.6638646125793457])

scale_x, scale_y = 0.5, 0.5
K_rgb = K_rgb_full.copy()
K_rgb[0, 0] *= scale_x
K_rgb[1, 1] *= scale_y
K_rgb[0, 2] *= scale_x
K_rgb[1, 2] *= scale_y

# ==== Extrinsics: depth to RGB (inverted) ====
q = [0.0488315560101656, 0.0003209717389778205, 0.0034281581399996505, 0.9988010863820517]
translation = [0.03215164832062954, 0.0026993268087001157, -0.0039073853888447925]
R = quat.quat2mat([q[3], q[0], q[1], q[2]])  # w, x, y, z
T_original = np.eye(4)
T_original[:3, :3] = R
T_original[:3, 3] = translation

# Invert the transformation matrix
T = np.linalg.inv(T_original)

# ==== Parameters ====
depth_scale = 0.001
rgb_shape = (768, 1024)

# ==== Load associations ====
pairs = []
with open(assoc_path, 'r') as f:
    for line in f:
        if line.strip() == '' or line.startswith('#'):
            continue
        tokens = line.strip().split()
        if len(tokens) >= 4:
            timestamp = tokens[0]
            rgb_path = os.path.join(base_dir, tokens[1])
            depth_path = os.path.join(base_dir, tokens[3])
            pairs.append((timestamp, rgb_path, depth_path))

# ==== Precompute Depth Undistort Map ====
h_d, w_d = 512, 512
map1_d, map2_d = cv2.initUndistortRectifyMap(K_d, D_d, None, K_d, (w_d, h_d), cv2.CV_32FC1)

# ==== Main Loop ====
for timestamp, rgb_path, depth_path in pairs:
    if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
        print(f"[WARN] Missing files for timestamp {timestamp}, skipping.")
        continue
    
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb_resized = cv2.imread(rgb_path)  # Keep original RGB (no undistortion)
    
    if depth_raw.shape != (512, 512) or rgb_resized.shape[:2] != rgb_shape:
        print(f"[WARN] Unexpected shape at {timestamp}, skipping.")
        continue
    
    # === Undistort depth only ===
    depth = cv2.remap(depth_raw, map1_d, map2_d, interpolation=cv2.INTER_NEAREST)
    
    # === Vectorized projection with RGB distortion ===
    aligned = np.zeros(rgb_shape, dtype=np.uint16)
    u_grid, v_grid = np.meshgrid(np.arange(w_d), np.arange(h_d))
    z = depth.astype(np.float32) * depth_scale
    valid = z > 0
    
    u = u_grid[valid]
    v = v_grid[valid]
    z = z[valid]
    
    x = (u - K_d[0, 2]) * z / K_d[0, 0]
    y = (v - K_d[1, 2]) * z / K_d[1, 1]
    points_d = np.stack((x, y, z, np.ones_like(z)), axis=1)
    
    points_rgb = (T @ points_d.T).T
    x_c, y_c, z_c = points_rgb[:, 0], points_rgb[:, 1], points_rgb[:, 2]
    
    # Project to normalized coordinates
    x_norm = x_c / z_c
    y_norm = y_c / z_c
    
    # Apply RGB distortion
    x_distorted, y_distorted = apply_distortion(x_norm, y_norm, D_rgb)
    
    # Convert to pixel coordinates
    u_rgb = (K_rgb[0, 0] * x_distorted + K_rgb[0, 2]).astype(np.int32)
    v_rgb = (K_rgb[1, 1] * y_distorted + K_rgb[1, 2]).astype(np.int32)
    
    valid_proj = (z_c > 0) & (u_rgb >= 0) & (u_rgb < rgb_shape[1]) & (v_rgb >= 0) & (v_rgb < rgb_shape[0])
    aligned[v_rgb[valid_proj], u_rgb[valid_proj]] = (z_c[valid_proj] / depth_scale).astype(np.uint16)
    
    out_path = os.path.join(output_dir, f"{timestamp}.png")
    cv2.imwrite(out_path, aligned)
    print(f"[INFO] Saved aligned depth map for timestamp {timestamp} -> {out_path}")