import numpy as np
import logging
import os
import sys
from glob import glob
import torch
import monai
from monai.data import Dataset, DataLoader,CacheDataset
from monai.inferers import sliding_window_inference
from PIL import Image
import json
from monai.data.utils import partition_dataset
from torchsummary import summary
from torch import nn
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandRotate90d,
    RandFlipd,
    ScaleIntensityd,
    ToTensord,
    RandSpatialCropSamplesd,
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import matplotlib.pyplot as plt
import time
import argparse



class PonderatedDiceloss(nn.Module):
    """PonderatedDiceloss for binary classification
     Shape:
        - Input: b * H * W * Z
        - Target:b * H * W * Z
    """

    def __init__(self) -> None:
        super(PonderatedDiceloss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
            ) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.shape == target.shape:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))

        intersection_1 = torch.sum(input * target, dim=list(range(1, input.dim())))
        union_1 = torch.sum(input, dim=list(range(1, input.dim()))) + torch.sum(target,
                                                                                dim=list(range(1, target.dim())))
        dice_1 = torch.mean(1.0 - (2. * intersection_1 + self.eps) / (union_1 + self.eps))

        target_2 = target * mask
        intersection_2 = torch.sum(input * mask * target_2, dim=list(range(1, input.dim())))
        union_2 = torch.sum(input * mask, dim=list(range(1, input.dim()))) + torch.sum(target_2,
                                                                                dim=list(range(1, target_2.dim())))
        dice_2 = torch.mean(1.0 - (2. * intersection_2 + self.eps) / (union_2 + self.eps))
        ponderated_dice = dice_1 + dice_2
        return ponderated_dice, dice_1, dice_2





def training(name_directory, name_dir_model, type_training, norm, roi_size=(96, 96), lr=1e-3, batch_size=32, max_epochs = 1000):
    """
    :param name_directory: path to the training dataset directory
    :name_dir_model: path to the directory where the model will be saved. the directory will be created
    :param type_training: three different training possible : "reconnect" (only reconnect and do not denoise), "reconnect_denoise"(reconnect and denoise), "denoise" (only denoise and do not reconnect)
    :param norm: norm used in the model
    :param roi_size:
    :param lr:
    :param batch_size:
    :param max_epochs:

    return
    """
    images = sorted(glob(f"{name_directory}/img_*.png"))
    loss_function = PonderatedDiceloss()

    if type_training == "reconnect":
        gts = sorted(glob(f"{name_directory}/seg_*.png"))
    elif type_training == "reconnect_denoise":
        gts = sorted(glob(f"{name_directory}/label_*.png"))
    elif type_training == "denoise":
        gts = sorted(glob(f"{name_directory}/denoise_deconnected_*.png"))

    mask_file = sorted(glob(f"{name_directory}/pos_*.png"))

    images_num = range(len(images))
    shuffle = True
    list_partition = partition_dataset(images_num, ratios=[4, 1], shuffle=shuffle)
    images_train = [images[x] for x in list_partition[0]]
    gts_train = [gts[x] for x in list_partition[0]]
    mask_train = [mask_file[x] for x in list_partition[0]]

    images_val = [images[x] for x in list_partition[1]]
    gts_val = [gts[x] for x in list_partition[1]]
    mask_val = [mask_file[x] for x in list_partition[1]]

    # for i, j, k in zip(images_val, gts_val, mask_val):
    #     print(i)
    #     print(j)
    #     print(k)
    #     print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=(norm),

    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr)
    os.mkdir(name_dir_model)

    training_loss = []
    training_loss_dice = []
    training_loss_dice_frag = []
    validation_loss = []
    validation_loss_dice = []
    validation_loss_dice_frag = []

    val_interval = 2

    train_files = [{"source_2D": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_train, gts_train, mask_train)]
    val_files = [{"source_2D": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_val, gts_val, mask_val)]

    train_trans = Compose(
        [
            LoadImaged(keys=["source_2D", "label", "mask"]),
            ScaleIntensityd(keys=["source_2D", "label", "mask"]),
            AddChanneld(keys=["source_2D", "label", "mask"]),
            RandSpatialCropSamplesd(keys=["source_2D", "label", "mask"], roi_size = roi_size, num_samples = 32, random_size=False),
            RandRotate90d(keys=["source_2D", "label", "mask"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["source_2D", "label", "mask"], prob=0.5, spatial_axis=[0, 1]),
            ToTensord(keys=["source_2D", "label", "mask"]),
        ]
        )

    val_trans = Compose(
        [
            LoadImaged(keys=["source_2D", "label", "mask"]),
            ScaleIntensityd(keys=["source_2D", "label", "mask"]),
            AddChanneld(keys=["source_2D", "label", "mask"]),
            ToTensord(keys=["source_2D", "label", "mask"]),
        ]
        )
    # define array dataset, data loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ################################# datasets ####################################################
    check_ds = CacheDataset(data = train_files, transform=train_trans)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available(), shuffle=True)

    val_ds = CacheDataset(data = val_files, transform=val_trans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    best_metric = 1000
    best_metric_epoch = -1
    metric_values = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        epoch_loss_norm_dice = 0
        epoch_loss_frag = 0
        step = 0
        for batch_data in check_loader:
            step += 1
            inputs, labels, masks = (
                batch_data["source_2D"].to(device),
                batch_data["label"].to(device),
                batch_data["mask"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss, dice, frag_dice = loss_function(outputs, labels, masks)
            if type_training == "denoise":
                loss_value = dice.item()
                dice.backward()
                optimizer.step()
                epoch_loss += dice.item()
            else:
                loss.backward()
                optimizer.step()
                loss_value = loss.item()
                epoch_loss += loss.item()
                epoch_loss_norm_dice += dice.item()
                epoch_loss_frag += frag_dice.item()
            print(
                f"{step}/{len(check_loader) // check_loader.batch_size}, "
                f"train_loss: {loss_value:.4f}")
        torch.save(model.state_dict(), os.path.join(name_dir_model, "last_model.pth"))
        print("step : ", step)
        epoch_loss /= step
        training_loss.append(epoch_loss)
        epoch_loss_norm_dice /= step
        training_loss_dice.append(epoch_loss_norm_dice)
        epoch_loss_frag /= step
        training_loss_dice_frag.append(epoch_loss_frag)

        metric_sum = 0.0
        metric_count = 0
        epoch_loss_D = 0
        epoch_loss_norm_dice = 0
        epoch_loss_frag = 0
        model.eval()
        with torch.no_grad():

            for (i, val_data) in enumerate(val_loader):
                val_inputs, val_labels,val_masks = (
                    val_data["source_2D"].to(device),
                    val_data["label"].to(device),
                    val_data["mask"].to(device)
                )
                sw_batch_size = 1
                val_outputs = sliding_window_inference(
                    val_inputs.float(), roi_size, sw_batch_size, model)
                val_outputs = torch.sigmoid(val_outputs)
                value1, val_norm_dice, val_dice_frag = loss_function(val_outputs, val_labels, val_masks)

                if type_training == "denoise":
                    epoch_loss_D += val_norm_dice.item()
                    metric_count +=val_outputs.shape[0]
                    metric_sum += val_norm_dice.sum().item()
                else:
                    epoch_loss_D += value1.item()
                    epoch_loss_norm_dice += val_norm_dice.item()
                    epoch_loss_frag += val_dice_frag.item()
                    metric_count +=val_outputs.shape[0]
                    metric_sum += value1.sum().item()


            metric = metric_sum / metric_count
            metric_values.append(metric)
            epoch_loss_D /= metric_count
            epoch_loss_norm_dice /= metric_count
            epoch_loss_frag /= metric_count

            validation_loss.append(epoch_loss_D)
            validation_loss_dice.append(epoch_loss_norm_dice)
            validation_loss_dice_frag.append(epoch_loss_frag)

            if metric < best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    name_dir_model, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
    torch.save(model.state_dict(), os.path.join(name_dir_model, "last_model.pth"))

    # global loss
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss
    plt.plot(x, y, "-", label="global Loss")
    torch.save((x, y), os.path.join(name_dir_model, "Dice_trainingLoss.pth"))

    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss
    plt.xlabel("epoch")
    plt.ylim(0, 1)

    plt.plot(x, y, ":", label="Global Validation Loss")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(name_dir_model + "/PDdice_training.png")
    # torch.save((x, y), os.path.join(name_dir_model, "Dice_validationLoss.pth"))
    plt.close()


    # normal dice loss
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss_dice
    plt.plot(x, y, "-", label="norm_dice_Training loss")
    # torch.save((x, y), os.path.join(name_dir_model, "normdice_trainingLoss.pth"))

    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss_dice
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.plot(x, y, ":", label="normdice_Validation Loss")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(name_dir_model + "/normal_dice_training.png")
    # torch.save((x, y), os.path.join(name_dir_model, "normdice_validationLoss.pth"))
    plt.close()

    # fragment dice loss
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss_dice_frag
    plt.plot(x, y, "-", label="frag_dice_Training Loss")

    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss_dice_frag
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.plot(x, y, ":", label="frag_dice__Validation Loss")


    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(name_dir_model + "/frag_dice_training.png")

    plt.close()


    training_config = {
        "dataset " : name_directory,
        "optimizer": "adam",
        "type_data":type_training,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": max_epochs,
        "loss": "PDdice",
        "norm": norm,
        "patch_size": roi_size,
        "best_epoch": best_metric_epoch,
        "best_diceloss": best_metric,
    }
    # Save config in json file
    with open( f'{name_dir_model}/config_training.json', 'w') as outfile:
        json.dump(training_config, outfile)
