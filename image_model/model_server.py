import csv

from typing import TypedDict
from flask_ml.flask_ml_server.models import DirectoryInput, ResponseBody, FileResponse, TaskSchema, InputSchema, ParameterSchema, InputType, TextParameterDescriptor
from flask_ml.flask_ml_server import MLServer


import torch
from sim_data import defaultDataset
# from BNN github
import model as model



def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output file",
        input_type=InputType.DIRECTORY
    )
    parameter_schema = ParameterSchema(
        key="ckpt_path",
        label="Path to the model checkpoint",
        value=TextParameterDescriptor(default="weights/dffd_M_unfrozen.ckpt")
    )
    return TaskSchema(
        inputs = [input_schema, output_schema],
        parameters = [parameter_schema]
    )


class Inputs(TypedDict):
    input_dataset: DirectoryInput
    output_file: DirectoryInput

class Parameters(TypedDict):
    ckpt_path: str

cfg = {
        "dataset_path": "datasets/demo",
        "resolution": 224,
        "ckpt": "weights/dffd_M_unfrozen.ckpt",
        }
server = MLServer(__name__)

@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs:Inputs, parameters:Parameters) -> ResponseBody:
    cfg["dataset_path"] = inputs["input_dataset"].path
    out = inputs["output_file"].path
    data = defaultDataset(
        dataset_path=cfg["dataset_path"],
        resolution=cfg["resolution"]
    )
    net = model.BNext4DFR.load_from_checkpoint(cfg["ckpt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    res_list = []
    for i in range(len(data)):
        sample = data[i]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = net(image)
            logit = output["logits"][0][0]
            temp = torch.sigmoid(logit)
            pred = 1 if temp > 0.9 else 0
            res_list.append({"image_path": sample["image_path"][:-1], "prediction": "real" if pred == 1 else "fake"})

    with open(out, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "prediction"])
        writer.writeheader()  # Write header row
        writer.writerows(res_list)  # Write data rows


    return ResponseBody(FileResponse(path=out, file_type="csv"))


if __name__ == "__main__":
    server.run(port=5000)