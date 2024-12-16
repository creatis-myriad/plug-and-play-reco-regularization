import logging
import os
import sys
import torch
import monai
import matplotlib.pyplot as plt
import json
import argparse
import random
from glob import glob
from monai.data.utils import partition_dataset
from monai.data import DataLoader, CacheDataset
from torch import nn
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandRotate90d,
    RandFlipd,
    ScaleIntensityd,
    ToTensord,
    RandSpatialCropd,
    CenterSpatialCropd,
    SpatialPadD
)


class PonderatedDiceloss(nn.Module):
    """Criterion Precision loss for binary classification

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
        dice = dice_1 + dice_2
        return dice, dice_1, dice_2




def training(name_directory, name_dir_model, type_training, norm, roi_size=(96, 96, 96), lr=1e-3, batch_size=4, max_epochs = 1000):



    os.mkdir(name_dir_model)
    ######################################### Parameters############################################################
    size_patch = roi_size
    shuffle = True
    seed = random.randint(0, 1000000000)
    images = sorted(glob(f"{name_directory}/img_*.nii.gz"))
    if type_training == "reconnect":
        gts = sorted(glob(f"{name_directory}/seg_*.nii.gz"))
    elif type_training == "reconnect_denoise":
        gts = sorted(glob(f"{name_directory}/label_*.nii.gz"))
    elif type_training == "denoise":
        gts = sorted(glob(f"{name_directory}/denoise_deconnected_*.nii.gz"))

    mask = sorted(glob(f"{name_directory}/pos_*.nii.gz"))
    ##################################### determination de la loss ################################################

    loss_function = PonderatedDiceloss()
    training_loss = []
    training_loss_dice = []
    training_loss_dice_frag = []
    validation_loss = []
    validation_loss_dice = []
    validation_loss_dice_frag = []


    #random training that can be reproduce
    monai.utils.set_determinism(seed=seed, additional_settings=None)

    # adding empty patches as there are not empty patches in vascusynth creating by adding artefacts to empty source_3D
    # images += sorted(glob(os.path.join(f"patch_vide/noise*.nii.gz")))
    # gts += sorted(glob(os.path.join(f"patch_vide/pos_deco*.nii.gz")))
    # mask += sorted(glob(f"patch_vide/pos_*.nii.gz"))


    images_num = range(len(images))
    list_partition = partition_dataset(images_num, ratios=[4, 1], shuffle=shuffle)

    # creation of the training lists
    images_train = [images[x] for x in list_partition[0]]
    gts_train = [gts[x] for x in list_partition[0]]
    masks_train = [mask[x] for x in list_partition[0]]

    images_val = [images[x] for x in list_partition[1]]
    gts_val = [gts[x] for x in list_partition[1]]
    masks_val = [mask[x] for x in list_partition[1]]

    # creation of training dictionnaries

    train_files = [{"image": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_train, gts_train, masks_train)]
    val_files = [{"image": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_val, gts_val, masks_val)]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model unet
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=(norm)
    ).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)


    train_trans = Compose(
        [
            LoadImaged(keys=["image", "label", "mask"]),
            ScaleIntensityd(keys=["image", "label", "mask"]),
            EnsureChannelFirstd(keys=["image", "label", "mask"]),
            SpatialPadD(keys=["image", "label", "mask"], spatial_size=[96,96,96]),
            RandRotate90d(keys=["image", "label", "mask"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["image", "label", "mask"], prob=0.5, spatial_axis=[0, 1]),
            RandSpatialCropd(["image", "label", "mask"], roi_size=size_patch, random_size=False),
            ToTensord(keys=["image", "label", "mask"]),
        ]
    )

    val_trans = Compose(
        [
            LoadImaged(keys=["image", "label", "mask"]),
            ScaleIntensityd(keys=["image", "label", "mask"]),
            EnsureChannelFirstd(keys=["image", "label", "mask"]),
            SpatialPadD(keys=["image", "label", "mask"], spatial_size=[96, 96, 96]),
            RandSpatialCropd(["image", "label", "mask"], roi_size=size_patch, random_size=False),
            ToTensord(keys=["image", "label", "mask"]),
        ]
    )

    # creation of datasets
    check_ds = CacheDataset(data=train_files, transform=train_trans)
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available(), shuffle=shuffle)

    val_ds = CacheDataset(data=val_files, transform=val_trans)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())


    best_metric = 100000
    best_metric_epoch = -1

    # training loop
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss_D = 0
        epoch_loss_norm_dice = 0
        epoch_loss_frag = 0
        step = 0
        for batch_data in check_loader:
            step += 1

            inputs, labels, masks = (batch_data["image"].to(device), batch_data["label"].to(device), batch_data["mask"].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss, dice_norm, dice_frag = loss_function(outputs, labels, masks)
            epoch_loss_D += loss.item()
            epoch_loss_norm_dice += dice_norm.item()
            epoch_loss_frag += dice_frag.item()

            loss.backward()
            optimizer.step()
            # epoch_loss += loss.item()
            print(
                f"{step}/{len(check_loader)}, "
                f"train_loss: {loss.item():.4f}")
        torch.save(model.state_dict(), os.path.join(name_dir_model, "last_model.pth"))
        print("step : ", step)

        epoch_loss_D /= step
        training_loss.append(epoch_loss_D)
        epoch_loss_norm_dice /= step
        training_loss_dice.append(epoch_loss_norm_dice)
        epoch_loss_frag /= step
        training_loss_dice_frag.append(epoch_loss_frag)

        model.eval()
        epoch_loss_D = 0
        epoch_loss_norm_dice = 0
        epoch_loss_frag = 0
        with torch.no_grad():
            metric_count = 0
            for val_data in val_loader:
                metric_count += 1
                # evaluation sur les patches
                val_inputs, val_labels, val_masks = (val_data["image"].to(device), val_data["label"].to(device), val_data["mask"].to(device) )
                val_outputs = model(val_inputs)
                val_outputs = torch.sigmoid(val_outputs)
                value1, val_norm_dice, val_dice_frag = loss_function(val_outputs, val_labels, val_masks)
                epoch_loss_D += value1.item()
                epoch_loss_norm_dice += val_norm_dice.item()
                epoch_loss_frag += val_dice_frag.item()

                epoch_loss_D /= metric_count
                epoch_loss_norm_dice /= metric_count
                epoch_loss_frag /= metric_count

                validation_loss.append(epoch_loss_D)
                validation_loss_dice.append(epoch_loss_norm_dice)
                validation_loss_dice_frag.append(epoch_loss_frag)

            metric = epoch_loss_D

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
    ## to decomment if you have enough data
    # plt.figure("train", (12, 6))
    # #dice loss
    # x = [i + 1 for i in range(max_epochs)]
    # y = training_loss
    # plt.plot(x, y, "-", label="D_Training Loss")
    # torch.save((x, y), os.path.join(name_dir_model, "Dice_trainingLoss.pth"))
    #
    # x = [i + 1 for i in range(max_epochs)]
    # y = validation_loss
    # plt.xlabel("epoch")
    # plt.ylim(0, 1)
    # plt.plot(x, y, ":", label="D_Validation Loss")
    #
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig(name_dir_model + "/dice_training.png")
    # torch.save((x, y), os.path.join(name_dir_model, "Dice_validationLoss.pth"))
    # torch.save((x, y), os.path.join(name_dir_model, "Dice_trainingLoss.pth"))
    # plt.close()
    # # BCE loss
    # x = [i + 1 for i in range(max_epochs)]
    # y = training_loss_dice
    # plt.plot(x, y, "-", label="BCE_Training Loss")
    # torch.save((x, y), os.path.join(name_dir_model, "BCE_trainingLoss.pth"))
    #
    # x = [i + 1 for i in range(max_epochs)]
    # y = validation_loss_dice
    # plt.xlabel("epoch")
    # plt.ylim(0, 1)
    # plt.plot(x, y, ":", label="BCE_Validation Loss")
    #
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig(name_dir_model + "/BCE_training.png")
    # torch.save((x, y), os.path.join(name_dir_model, "BCE_validationLoss.pth"))
    # torch.save((x, y), os.path.join(name_dir_model, "BCE_trainingLoss.pth"))
    # plt.close()
    # # DBCE loss
    # x = [i + 1 for i in range(max_epochs)]
    # y = training_loss_dice_frag
    # plt.plot(x, y, "-", label="DBCE_Training Loss")
    #
    # x = [i + 1 for i in range(max_epochs)]
    # y = validation_loss_dice_frag
    # plt.xlabel("epoch")
    # plt.ylim(0, 1)
    # plt.plot(x, y, ":", label="DBCE_Validation Loss")
    #
    #
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig(name_dir_model + "/DBCE_training.png")
    #
    # torch.save((x, y), os.path.join(name_dir_model, "DBCE_validationLoss.pth"))
    # torch.save((x, y), os.path.join(name_dir_model, "DBCE_trainingLoss.pth"))
    # plt.close()
    # # BCE loss et Dice loss séparées
    # x = [i + 1 for i in range(max_epochs)]
    # y = training_loss
    # plt.plot(x, y, "-", label="Dice Loss")
    #
    # x = [i + 1 for i in range(max_epochs)]
    # y = validation_loss_dice
    # plt.xlabel("epoch")
    # plt.ylim(0, 1)
    # plt.plot(x, y, ":", label="BCE Loss")
    #
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig(name_dir_model + "/DBCE_separe_training.png")
    # plt.close()

    training_config = {
        "type_training": type_training,
        "optimizer": "adam",
        "norm": norm,
        "patch_vide": True,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": max_epochs,
        "patch_size": size_patch,
        "best_epoch": best_metric_epoch,
        "best_diceloss": best_metric,
        "seed": seed,
    }
    # Save config in json file
    with open( f'{name_dir_model}/config_training.json', 'w') as outfile:
        json.dump(training_config, outfile)
