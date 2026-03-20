from DynamicAwareTransformer import creat_agrifm_base, creat_agrifm_tiny
import os
import os.path as osp

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

import torch

import torchvision.transforms as transforms


def load_weights_from_ckpt_model(model, ckpt_path):

    print(f"Loading weights from {ckpt_path}")

    if os.path.exists(ckpt_path):

        pass
    else:
        print(f"Warning: Checkpoint file {ckpt_path} not found!")


def process_single_image_AgriFMB(image_path, save_fg_mask=False, img_size=224, output_folder="PCA_outputs"):
    os.makedirs(output_folder, exist_ok=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = creat_agrifm_base(123)
    pretraied_weight = r""
    load_weights_from_ckpt_model(model, pretraied_weight)
    model.to(device)
    model.eval()

    # --- 修改点 2: 图像预处理 ---
    processor = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


    image = Image.open(image_path).convert('RGB')
    image = processor(image).unsqueeze(0).to(device)

    image_plot = ((image.cpu().numpy() * np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) * 255).transpose(0, 2, 3, 1).astype(np.uint8)[0]


    with torch.no_grad():
        outputs, layer_outputs = model(image)
        hidden_states = layer_outputs[-1]
        patch_features = hidden_states[0].cpu().numpy()

    print(f"Extracted features shape: {patch_features.shape}")


    num_patches = patch_features.shape[0]
    patch_h = patch_w = int(np.sqrt(num_patches))


    x_norm_1616_patches = patch_features.reshape(1, patch_h * patch_w, -1)

    fg_pca = PCA(n_components=1)
    fg_pca_images = fg_pca.fit_transform(x_norm_1616_patches.reshape(-1, x_norm_1616_patches.shape[-1]))
    fg_pca_images = minmax_scale(fg_pca_images)  # 归一化
    fg_pca_images = fg_pca_images.reshape(1, -1)


    mask = (fg_pca_images > 0.6).ravel()

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(x_norm_1616_patches.reshape(-1, x_norm_1616_patches.shape[-1]))
    pca_features = pca_features.reshape(1, -1, 3)
    pca_features[0, ~mask] = np.min(pca_features)
    pca_features = pca_features.reshape(1, -1, 3)
    pca_features = pca_features.squeeze(0)
    fg_result = minmax_scale(pca_features)
    fg_result = fg_result.reshape(1, patch_h, patch_w, 3)


    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(fg_result[0])

    plt.savefig(osp.join(output_folder, "AgriFM-B.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close()

    return fg_result[0]


if __name__ == "__main__":
    image_path = r""
    save_fg_mask = True
    img_size = 224
    output_folder = "outputs_AgriFM-T"
    process_single_image_AgriFMB(image_path, save_fg_mask, img_size, output_folder)