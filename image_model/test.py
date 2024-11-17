import argparse

from torchmetrics.functional.classification import accuracy
import torch
from retinaface import RetinaFace
from sim_data import defaultDataset
# from BNN github
import model as model
import matplotlib.pyplot as plt

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/demo",
        help="Path to the dataset folder.",
    )
    parser.add_argument(
        "--ckpt_path",  
        type=str,
        default="weights/dffd_M_unfrozen.ckpt",
        help="Path to the model checkpoint.",
    )
    args = parser.parse_args()
    return args

def get_area_ratio(img1, img2):
    print(img1, img2)
    h1, w1 = img1
    h2, w2 = img2
    return (h1 * w1) / (h2 * w2)


if __name__ == "__main__":
    
    cfg = {
        "dataset_path": "datasets/demo",
        "resolution": 224,
        "ckpt": "weights/dffd_M_unfrozen.ckpt",
        "enable_facecrop": True,
        }

    args = args_func()

    cfg["dataset_path"] = args.dataset_path
    cfg["ckpt"] = args.ckpt_path

    test_dataset = defaultDataset(
        dataset_path=cfg["dataset_path"],
        resolution=cfg["resolution"]
    )

    model_ckpt = cfg["ckpt"]
    net = model.BNext4DFR.load_from_checkpoint(model_ckpt)
    print(f"Model loaded from {model_ckpt}")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)


    # run_label = args.cfg.split("/")[-1].split(".")[0]
    label = []
    logits = []

    for i in range(len(test_dataset))[:10000]:
        sample = test_dataset[i]
        image = None
        if cfg["enable_facecrop"]:
                faces = RetinaFace.extract_faces(sample["image_path"][:-1], expand_face_area=35)       
                if len(faces) > 0:
                    image = test_dataset.apply_transforms(faces[0])
                else:
                    image = sample["image"]
        else:
            image = sample["image"]
        image = image.unsqueeze(0).to(device)
        is_real = sample["is_real"].to(device)
        with torch.no_grad():
            output = net(image)
            logit = output["logits"][0][0]
            temp = torch.sigmoid(logit)
            pred = 1 if temp > 0.9 else 0
            print(f"Image: {sample['image_path']}")
            print(temp)
            print(f"Predicted: {pred}, Real: {is_real[0]}")
            logits.append(logit)
            label.append(is_real[0])
        
    acc = accuracy(preds=torch.stack(logits), target=torch.stack(label), task="binary", average="micro", threshold=0.9)
    print(f"Accuracy: {acc}")
