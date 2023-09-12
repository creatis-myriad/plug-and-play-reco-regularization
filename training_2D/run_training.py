import logging
import os
import sys
from glob import glob
import torch
import monai
from monai.data import DataLoader,CacheDataset
from monai.inferers import sliding_window_inference
import json
from monai.data.utils import partition_dataset
from personnal_transforms import PonderatedDiceloss
from torchsummary import summary

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

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m-%d-%Y_%H-%M", named_tuple)
parser = argparse.ArgumentParser(description='process parameters of the training')
parser.add_argument('taille_deco_max', type=str, default="1", help='numero de limage traite entre 1 et 40')
parser.add_argument('type_data', type=str, default="reconnect", help='path to the model')
parser.add_argument('norm', type=str, default="instance", help='path to the model')

max_epochs = 1000

args = parser.parse_args()
taille_deco_max = args.taille_deco_max
norm = args.norm
########################################### Parameters########################################################""
# pourcent_data = 0.7
type_data = args.type_data
if taille_deco_max != "all":
    img_file = f"dataset_{taille_deco_max}/deconnexions"
else:
    img_file = f"dataset_*/deconnexions"
images = sorted(glob(os.path.join(img_file, "img_*.png")))

loss_function = PonderatedDiceloss()

if type_data == "reconnect":
    gts = sorted(glob(os.path.join(img_file, "seg_*.png")))
elif type_data == "reconnect_denoise":
    gts = sorted(glob(os.path.join(img_file, "label_*.png")))
elif type_data == "fragment":
    gts = sorted(glob(os.path.join(img_file, "deco_*.png")))
elif type_data == "denoise":
    gts = sorted(glob(os.path.join(img_file, "denoise_deconnected_*.png")))

mask_file = sorted(glob(os.path.join(img_file, "pos_deco_*.png")))

images_num = range(len(images))
shuffle = True
list_partition = partition_dataset(images_num, ratios=[4, 1], shuffle=shuffle)
images_train = [images[x] for x in list_partition[0]]
gts_train = [gts[x] for x in list_partition[0]]
mask_train = [mask_file[x] for x in list_partition[0]]

images_val = [images[x] for x in list_partition[1]]
gts_val = [gts[x] for x in list_partition[1]]
mask_val = [mask_file[x] for x in list_partition[1]]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=(norm),

).to(device)
summary(model, input_size=(1, 96, 96))


nb_image_train = len(images_train)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
root_dir = f"modele_2D/{time_string}/"
os.mkdir(root_dir)

loss_function = PonderatedDiceloss()
training_loss = []
training_loss_dice = []
training_loss_dice_frag = []
validation_loss = []
validation_loss_dice = []
validation_loss_dice_frag = []

val_interval = 2

train_files = [{"image": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_train, gts_train, mask_train)]
val_files = [{"image": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_val, gts_val, mask_val)]

print(len(train_files))
roi_size = (96, 96)
train_trans = Compose(
    [
        LoadImaged(keys=["image", "label", "mask"]),
        ScaleIntensityd(keys=["image", "label", "mask"]),
        AddChanneld(keys=["image", "label", "mask"]),
        RandSpatialCropSamplesd(keys=["image", "label", "mask"], roi_size = roi_size, num_samples = 32, random_size=False),
        RandRotate90d(keys=["image", "label", "mask"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label", "mask"], prob=0.5, spatial_axis=[0, 1]),
        ToTensord(keys=["image", "label", "mask"]),
    ]
    )

val_trans = Compose(
    [
        LoadImaged(keys=["image", "label", "mask"]),
        ScaleIntensityd(keys=["image", "label", "mask"]),
        AddChanneld(keys=["image", "label", "mask"]),
        ToTensord(keys=["image", "label", "mask"]),
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
epoch_loss_values = []
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
            batch_data["image"].to(device),
            batch_data["label"].to(device),
            batch_data["mask"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        loss, dice, frag_dice = loss_function(outputs, labels, masks)
        if type_data == "denoise":
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
    torch.save(model.state_dict(), os.path.join(root_dir, "last_model.pth"))
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
                val_data["image"].to(device),
                val_data["label"].to(device),
                val_data["mask"].to(device)
            )
            sw_batch_size = 1
            val_outputs = sliding_window_inference(
                val_inputs.float(), roi_size, sw_batch_size, model)
            val_outputs = torch.sigmoid(val_outputs)
            value1, val_norm_dice, val_dice_frag = loss_function(val_outputs, val_labels, val_masks)
            print(value1, val_norm_dice, val_dice_frag)
            if type_data == "denoise":
                epoch_loss_D += val_norm_dice.item()
                metric_count +=val_outputs.shape[0]
                metric_sum += val_norm_dice.sum().item()
                val_outputs = (val_outputs > 0.5) * 255
            else:
                epoch_loss_D += value1.item()
                epoch_loss_norm_dice += val_norm_dice.item()
                epoch_loss_frag += val_dice_frag.item()
                metric_count +=val_outputs.shape[0]
                metric_sum += value1.sum().item()
                val_outputs = (val_outputs > 0.5) * 255


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
                root_dir, "best_metric_model.pth"))
            print("saved new best metric model")
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f"\nbest mean dice: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )
torch.save(model.state_dict(), os.path.join(root_dir, "last_model.pth"))


# global loss
x = [i + 1 for i in range(max_epochs)]
y = training_loss
plt.plot(x, y, "-", label="global Loss")
torch.save((x, y), os.path.join(root_dir, "Dice_trainingLoss.pth"))

x = [i + 1 for i in range(max_epochs)]
y = validation_loss
plt.xlabel("epoch")
plt.ylim(0, 1)

plt.plot(x, y, ":", label="Global Validation Loss")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(root_dir + "/dice_training.png")
torch.save((x, y), os.path.join(root_dir, "Dice_validationLoss.pth"))
plt.close()


# normal dice loss
x = [i + 1 for i in range(max_epochs)]
y = training_loss_dice
plt.plot(x, y, "-", label="norm_dice_Training loss")
torch.save((x, y), os.path.join(root_dir, "normdice_trainingLoss.pth"))

x = [i + 1 for i in range(max_epochs)]
y = validation_loss_dice
plt.xlabel("epoch")
plt.ylim(0, 1)
plt.plot(x, y, ":", label="normdice_Validation Loss")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(root_dir + "/normdice_training.png")
torch.save((x, y), os.path.join(root_dir, "normdice_validationLoss.pth"))
plt.close()

# fragment dice loss
x = [i + 1 for i in range(max_epochs)]
y = training_loss_dice_frag
plt.plot(x, y, "-", label="frag_dice_Training Loss")
torch.save((x, y), os.path.join(root_dir, "frag_dice_trainingLoss.pth"))

x = [i + 1 for i in range(max_epochs)]
y = validation_loss_dice_frag
plt.xlabel("epoch")
plt.ylim(0, 1)
plt.plot(x, y, ":", label="DBCE_Validation Loss")
torch.save((x, y), os.path.join(root_dir, "frag_dice_validationLoss.pth"))


plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(root_dir + "/DBCE_training.png")

plt.close()



training_config = {
    "optimizer": "adam",
    "taille_deco_max": taille_deco_max,
    "type_data":type_data,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": max_epochs,
    "loss": "PDdice",
    "norm": norm,
    "patch_size": roi_size,
    "best_epoch": best_metric_epoch,
    "best_diceloss": best_metric,
}
# Save config in json file
with open( f'{root_dir}/config_training.json', 'w') as outfile:
    json.dump(training_config, outfile)
