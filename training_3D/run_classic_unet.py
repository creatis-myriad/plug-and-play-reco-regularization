import numpy as np
import logging
import os
import sys
from glob import glob
import torch
import monai
from monai.data import ArrayDataset, create_test_image_2d,PatchDataset, Dataset,DataLoader
from monai.inferers import sliding_window_inference

import matplotlib.pyplot as plt
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandRotate90d,
    RandFlipd,
    RandSpatialCropd,
    ScaleIntensityd,
    ToTensord,
    RandSpatialCropSamplesd,
    RandGaussianNoised,
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import argparse

########################################### Parameters########################################################""

img_file = "../../unet_dataset_STARE/images_gray"
images = sorted(glob(os.path.join(img_file, "img*.png")))
gt_file = "../../unet_dataset_STARE/GT"
gts = sorted(glob(os.path.join(gt_file, "seg*.png")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
model = monai.networks.nets.UNet(
    dimensions=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    num_res_units=0,
).to(device)


optimizer = torch.optim.Adam(model.parameters(), 1e-3)
# root_dir = args.model_path
root_dir = "../modele/unet"

loss_function = monai.losses.DiceLoss(sigmoid=True)
# loss_function = torch.nn.BCELoss()

max_epochs = 2000
val_interval = 2

train_files = [{"image": img, "label": gt} for img, gt in zip(images[:-2], gts[:-2])]
val_files = [{"image": img, "label": gt} for img, gt in zip(images[-2:], gts[-2:])]
print(len(train_files))
roi_size = (96, 96)
train_trans = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        RandSpatialCropSamplesd(keys=["image", "label"], roi_size = roi_size, num_samples = 4, random_size=False),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
        ToTensord(keys=["image", "label"]),
    ]
    )

val_trans = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ]
    )
# define array dataset, data loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ################################# datasets ####################################################
check_ds = Dataset(data = train_files, transform=train_trans)
check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

val_ds = Dataset(data = val_files, transform=val_trans)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
#
################################ Affichage d'images de vérification ##################################

for batch_data in check_loader:
    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
    print(inputs.shape)
    print("walla cest long")
    for i in range(inputs.shape[0]):
        plt.figure()
        plt.subplot(121)
        plt.imshow(inputs[i].squeeze(), cmap='gray')
        plt.subplot(122)
        plt.imshow(labels[i].squeeze(), cmap='gray')

        plt.show()
    break
# # # ######################################################################################################
best_metric = 100000
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in check_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(check_loader) // check_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(root_dir, "last_model.pth"))
    print("step : ", step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                sw_batch_size = 5
                val_outputs = sliding_window_inference(val_inputs.float(), roi_size, sw_batch_size, model)
                value = loss_function(val_outputs, val_labels)
                metric_count +=val_outputs.shape[0]
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric < best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
torch.save(model.state_dict(), os.path.join(root_dir, "last_model.pth"))
plt.figure("train", (12, 6))
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.plot(x, y, "-", label="Training Loss")
torch.save((x, y), os.path.join(root_dir, "trainingLoss.pth"))

x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.ylim(0, 1)
plt.plot(x, y, ":", label="Validation Loss")
torch.save((x, y), os.path.join(root_dir, "validationLoss.pth"))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(root_dir+"/evolution_training.png")
