"""Score all validation images with the saved ResNet18 model checkpoint on CPU.

Purpose:
    Performs deterministic CPU inference on the 431 independent validation images:
    1. Loads the fitted ResNet18 binary classifier weights (checkpoint state dict).
    2. Resizes images to 64x64 and applies standard ImageNet normalization.
    3. Computes softmax probabilities for class 0 (glamping) and class 1 (camping).
    4. Records predicted labels, class probabilities, confidence scores, and correctness flags.

Input Files:
    - retained/best_glamping.pt: PyTorch checkpoint of the trained ResNet18 binary classifier.
    - input/cnn_images/valid/: 431 validation images in subdirectories '0' (glamping) and '1' (camping).

Output Files:
    - output/cnn_validation/cnn_validation_predictions.csv: Detailed prediction results for all 431 validation images.
"""
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from common import DATA, RETAINED, OUT, out_path, require, save


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    checkpoint = torch.load(RETAINED / "best_glamping.pt", map_location="cpu",
                            weights_only=True)
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    paths = sorted((DATA / "cnn_images/valid").rglob("*.jpg"))
    manifest = pd.DataFrame({"image_path": [str(p.relative_to(DATA)) for p in paths],
                             "true_idx": [int(p.parent.name) for p in paths]})
    require(len(manifest) == 431 and manifest.image_path.is_unique,
            "Expected 431 unique validation images")
    rows = []
    with torch.inference_mode():
        for item in manifest.itertuples(index=False):
            relative = Path(item.image_path)
            require(relative.parts[:2] == ("cnn_images", "valid")
                    and relative.parts[2] in ("0", "1"), "Invalid image class path")
            true_idx = int(relative.parts[2])
            require(true_idx == int(item.true_idx), "Image path and class label differ")
            with Image.open(DATA / relative) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            probabilities = torch.softmax(net(tensor), dim=1)[0].tolist()
            predicted = int(max(range(2), key=lambda i: probabilities[i]))
            rows.append({"image_path": item.image_path, "true_idx": true_idx,
                         "true_label": "glamping" if true_idx == 0 else "camping",
                         "pred_idx": predicted,
                         "pred_label": "glamping" if predicted == 0 else "camping",
                         "prob_glamping": probabilities[0],
                         "prob_camping": probabilities[1],
                         "confidence": probabilities[predicted],
                         "correct": true_idx == predicted})
    save(pd.DataFrame(rows), "cnn_validation_predictions.csv")
    print(f"Scored {len(rows)} validation photographs on CPU.", flush=True)


if __name__ == "__main__":
    main()
