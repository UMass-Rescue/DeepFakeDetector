import csv

from typing import TypedDict
from flask_ml.flask_ml_server.models import DirectoryInput, ResponseBody, FileResponse, TaskSchema, InputSchema, ParameterSchema, InputType, EnumParameterDescriptor, TextParameterDescriptor, EnumVal
from flask_ml.flask_ml_server import MLServer

from retinaface import RetinaFace
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
    ckpt_schema = ParameterSchema(
        key="ckpt_path",
        label="Path to the model checkpoint",
        value=TextParameterDescriptor(default="weights/dffd_M_unfrozen.ckpt")
    )
    bool_schema = ParameterSchema(
        key="disable_facecrop",
        label="Disable facecrop",
        value=EnumParameterDescriptor(default="False", enum_vals=[EnumVal(key="True", label="True"), EnumVal(key="False", label="False")])
    )
    return TaskSchema(
        inputs = [input_schema, output_schema],
        parameters = [ckpt_schema, bool_schema]
    )


class Inputs(TypedDict):
    input_dataset: DirectoryInput
    output_file: DirectoryInput

class Parameters(TypedDict):
    ckpt_path: str
    disable_facecrop: str

cfg = {
        "dataset_path": "datasets/demo",
        "resolution": 224,
        "ckpt": "weights/dffd_M_unfrozen.ckpt",
        }
server = MLServer(__name__)


def predict(net, sample, device, dataset, disable_facecrop=False):
    image = None
    if not disable_facecrop:
            faces = RetinaFace.extract_faces(sample["image_path"][:-1], expand_face_area=35)       
            if len(faces) > 0:
                image = dataset.apply_transforms(faces[0])
            else:
                image = sample["image"]
    else:
        image = sample["image"]
    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = net(image)
        logit = output["logits"][0][0]
        temp = torch.sigmoid(logit)
        print(temp)
        pred = 1 if temp > 0.9 else 0
        return pred

@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs:Inputs, parameters:Parameters) -> ResponseBody:
    cfg["dataset_path"] = inputs["input_dataset"].path
    out = inputs["output_file"].path
    data = defaultDataset(
        dataset_path=cfg["dataset_path"],
        resolution=cfg["resolution"]
    )
    print(parameters)
    disable_facecrop = parameters["disable_facecrop"] == "True"
    net = model.BNext4DFR.load_from_checkpoint(cfg["ckpt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    res_list = []
    for i in range(len(data)):
        sample = data[i]
        pred = predict(net, sample, device, data, disable_facecrop)
        res_list.append({"image_path": sample["image_path"][:-1], "prediction": "real" if pred == 1 else "fake"})

    with open(out, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "prediction"])
        writer.writeheader()  # Write header row
        writer.writerows(res_list)  # Write data rows


    return ResponseBody(FileResponse(path=out, file_type="csv"))


if __name__ == "__main__":
    server.run(port=5000)