import argparse

from torchmetrics.functional.classification import accuracy
import torch

from sim_data import defaultDataset
# from BNN github
import model as model


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


if __name__ == "__main__":
    
    cfg = {
        "dataset_path": "datasets/demo",
        "resolution": 224,
        "ckpt": "weights/dffd_M_unfrozen.ckpt",
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
        image = sample["image"].unsqueeze(0).to(device)
        is_real = sample["is_real"].to(device)
        with torch.no_grad():
            output = net(image)
            print(output)
            logit = output["logits"][0][0]
            temp = torch.sigmoid(logit)
            print(temp)
            pred = 1 if temp > 0.9 else 0
            print(f"Image: {sample['image_path']}")
            print(f"Predicted: {pred}, Real: {is_real[0]}")
            logits.append(logit)
            label.append(is_real[0])
        
    acc = accuracy(preds=torch.stack(logits), target=torch.stack(label), task="binary", average="micro", threshold=0.9)
    print(f"Accuracy: {acc}")
