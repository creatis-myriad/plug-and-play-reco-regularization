import numpy as np
import torch
import monai
import json
from monai.inferers import sliding_window_inference
from sources import image_utils

def monai_predict_image(image, model, roi_size =(96, 96), sw_batch_size = 5, mode = "gaussian", overlap = 0.5, device="cpu"):
    """
    :param image: source_2D on which we infer with the model
    :param model: trained model
    :param roi_size: patch size (tuple)
    :param sw_batch_size: the batch size to run window slices.
    :param mode: How to blend output of overlapping windows. (from monai)
    :param overlap: Amount of overlap between scans.
    :param device: cpu or gpu
    return the source_2D infered with the model
    """
    image = torch.from_numpy(image)
    image = image.float().unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(image, roi_size, sw_batch_size, model, mode = mode, overlap = overlap)
    output = output.squeeze()
    output = torch.sigmoid(output).cpu().numpy()
    return output


def post_treatement(segmentation_path, model_directory_path, iterations=10):
    """
    Apply the model an iteration number of time on the source_2D stocked at the segmentation_path
    :param segmentation_path: source_2D on which we infer with the model
    :param model_directory_path: trained model
    :param iterations: patch size (tuple)

    return the post treated source_2D
    """
    model_file =f"{model_directory_path}/best_metric_model.pth"

    parameters_training = open(f"{model_directory_path}/config_training.json")
    parameters_training = json.load(parameters_training)
    norm = parameters_training["norm"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=(norm)
    ).to(device)

    if device == "cuda":
        model.load_state_dict(torch.load(model_file)).to(device)
    else:
        model.load_state_dict(torch.load(model_file, map_location="cpu"))

    image = image_utils.read_image(segmentation_path)
    image = ((image >= 0.5) * 255).astype(np.uint8)

    for i in range(1, iterations + 1):
        image = image_utils.normalize_image(image, 1)
        image = monai_predict_image(image, model, roi_size=parameters_training["patch_size"], device=device)
        image = ((image >= 0.5) * 255).astype(np.uint8)

    return image