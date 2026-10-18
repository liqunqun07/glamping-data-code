"""Generate Grad-CAM activation maps for validation images (Appendix Figure F2).

Purpose:
    Produces Gradient-weighted Class Activation Mapping (Grad-CAM) visualizations
    to explain visual features salient to the ResNet18 classifier:
    1. Attaches forward and backward hooks to the final residual block (model.layer3[-1]).
    2. Computes gradient-weighted feature maps for representative validation images.
    3. Overlays JET colormap heatmaps onto original photographs.
    4. Compares visual attention between Glamping (fixed structures) and Camping (standard tents).

Input Files:
    - retained/best_glamping.pt: Saved ResNet18 PyTorch model checkpoint.
    - input/cnn_images/valid/: Sample validation images for glamping (class 0) and camping (class 1).

Output Files:
    - output/figures/gradcam_examples.png: Side-by-side original and Grad-CAM overlay comparison figure (500 dpi).
"""
import os
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from common import DATA, RETAINED, OUT, out_path
BASE_DIR = str(DATA)
ckpt_path = str(RETAINED / "best_glamping.pt")
cat_to_name = {"0": "glamping", "1": "camping"}

device = torch.device("cpu")

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval().to(device)

activations = {}
gradients = {}

def fwd_hook(module, inp, out):
    activations["value"] = out.detach()

def bwd_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0].detach()

target_layer = model.layer3[-1]
target_layer.register_forward_hook(fwd_hook)
target_layer.register_full_backward_hook(bwd_hook)

infer_transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def gradcam_for_image(img_path):
    img = Image.open(img_path).convert("RGB")
    inp = infer_transform(img).unsqueeze(0).to(device)
    inp.requires_grad_(True)

    model.zero_grad()
    output = model(inp)
    pred_idx = output.argmax(dim=1).item()
    score = output[0, pred_idx]
    score.backward()

    acts = activations["value"][0]       # [C, H, W]
    grads = gradients["value"][0]        # [C, H, W]
    weights = grads.mean(dim=(1, 2))     # GAP over spatial dims -> [C]

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for c, w in enumerate(weights):
        cam += w * acts[c]
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.numpy()

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)
    cam_arr = np.array(cam_img).astype(float) / 255.0

    heatmap = cm.jet(cam_arr)[:, :, :3]
    img_arr = np.array(img).astype(float) / 255.0
    overlay = 0.55 * img_arr + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return img_arr, overlay, cat_to_name[str(pred_idx)]


samples = [
    ("cnn_images/valid/0/209.jpg", "glamping"),
    ("cnn_images/valid/0/99.jpg", "glamping"),
    ("cnn_images/valid/0/463.jpg", "glamping"),
    ("cnn_images/valid/1/54.jpg", "camping"),
    ("cnn_images/valid/1/60.jpg", "camping"),
    ("cnn_images/valid/1/223.jpg", "camping"),
]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

fig, axes = plt.subplots(2, len(samples), figsize=(3.2 * len(samples), 6.6))
for col, (rel_path, true_label) in enumerate(samples):
    full_path = os.path.join(BASE_DIR, rel_path)
    img_arr, overlay, pred_label = gradcam_for_image(full_path)

    ax0 = axes[0, col]
    ax0.imshow(img_arr)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title(f"true: {true_label}", fontsize=11)
    for s in ax0.spines.values():
        s.set_visible(True); s.set_color("black")

    ax1 = axes[1, col]
    ax1.imshow(overlay)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"Grad-CAM (pred: {pred_label})", fontsize=11)
    for s in ax1.spines.values():
        s.set_visible(True); s.set_color("black")

plt.tight_layout()
fig.savefig(out_path("gradcam_examples.png"), dpi=500, facecolor="white", bbox_inches="tight")
print("Saved gradcam_examples.png")
