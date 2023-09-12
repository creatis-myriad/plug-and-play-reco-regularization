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
from monai.data import DataLoader, write_nifti, CacheDataset
from personnal_transforms import PonderatedDiceloss
from torchsummary import summary
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandRotate90d,
    RandFlipd,
    ScaleIntensityd,
    ToTensord,
    RandSpatialCropd,
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# saisie des parametres de l'entrainement
parser = argparse.ArgumentParser(description='process parameters of the training')
parser.add_argument('dataset', type=str, default=1, help='dataset selectionné')
parser.add_argument('epoch', type=int, default=1, help='dataset selectionné')
parser.add_argument('lr', type=float, default=0.01, help='dataset selectionné')
parser.add_argument('norm', type=str, default="None", help='dataset selectionné')
parser.add_argument('batch_size', type=int, default=2, help='dataset selectionné')
parser.add_argument('taille_patch', type=int, default=96, help='dataset selectionné')
parser.add_argument('input_bruit', type=str, default="non_art_deco", help='dataset selectionné')
parser.add_argument('output_bruit', type=str, default="label", help='dataset selectionné')
parser.add_argument('noise_data', type=str, default="True", help='dataset selectionné')
parser.add_argument('loss', type=str, default="D", help='dataset selectionné')
parser.add_argument('residual_archi', type=str, default="True", help='dataset selectionné')

args = parser.parse_args()
######################################### Parameters############################################################
max_epochs = args.epoch
lr = args.lr
size_patch = (args.taille_patch, args.taille_patch, args.taille_patch)
# overlap = args.overlap
batch_size =args.batch_size
shuffle = True
seed = random.randint(0, 10000000000)
input_bruit = args.input_bruit
residual = (args.residual_archi =="True")
output_bruit = args.output_bruit
if args.norm == "None":
    norm = None
elif args.norm == "batch":
    norm =  "batch"
elif args.norm == "instance":
    norm = "instance"
else:
    print("mauvaise norme , tes mauvais frr")
    exit()
##################################### determination de la loss ################################################
determine_loss = args.loss
if determine_loss == "D":
    loss_function1 = monai.losses.DiceLoss(sigmoid=False)
    training_loss = []
    validation_loss = []
    beta = 1
elif determine_loss == "PD":
    loss_function1 = PonderatedDiceloss()
    training_loss = []
    training_loss_dice = []
    training_loss_dice_frag = []
    validation_loss = []
    validation_loss_dice = []
    validation_loss_dice_frag = []
    # loss_function1 = monai.losses.DiceCELoss(sigmoid=True, include_background=True)
elif determine_loss =="P":
    beta = 1
    loss_function1 = PrecisionLoss()
    training_loss = []
    validation_loss = []

#entrainement aléatoire mais avec une seed pour pouvoir la reproduire
monai.utils.set_determinism(seed=seed, additional_settings=None)
root_dir = "trainings"
if residual:
    name_resi = "residual_arch"
else:
    name_resi = "unet"
if args.noise_data == "True":
    noise_data = True
else:
    noise_data = False
print(name_resi)

name = f"{args.dataset}_{input_bruit}_{max_epochs}_{lr}_{name_resi}_{determine_loss}_{seed}"
# creation du dossier ou sauver les résultats
existed_trainings = sorted(os.listdir(root_dir))
if name in existed_trainings:
    print("deja fait, revoie ta notation")
    exit()
else:
    os.mkdir(f"{root_dir}/{name}")
    root_dir = f"{root_dir}/{name}"

# choix du dataset d'entrainement / bullit seul / vascu seul / bullit et vascu
if args.dataset == "Brava":
    img_file = "brava_deco"
elif args.dataset == "Brava_2":
    # img_file = "brava_deco_2"
    img_file = "test_brava_deco_2"

elif args.dataset == "vascu":
    # img_file = "new_vascusynth_deco_rad_4"
    img_file = "test_vascu"
elif args.dataset =="cco":
    img_file = "test_cco"

else:
    exit()

images = sorted(glob(os.path.join(img_file, f"{input_bruit}*.nii.gz")))
gts = sorted(glob(os.path.join(img_file, f"{output_bruit}*.nii.gz")))
if noise_data:
    images += sorted(glob(os.path.join(f"patch_vide/noise*.nii.gz")))
    # if output_bruit == "pos_deco":
    gts += sorted(glob(os.path.join(f"patch_vide/pos_deco*.nii.gz")))
    # else:
    #     gts += sorted(glob(os.path.join(f"patch_vide/*.nii.gz")))

# for i, j in zip(images, gts):
#     print(i)
#     print(j)
#     print()
# # partition du dataset
# exit()
print(len(images))
print(images)
images_num = range(len(images))
list_partition = partition_dataset(images_num, ratios=[4, 1], shuffle=shuffle)
print(list_partition)

# creation des listes de trains ou sont situés les patchs
images_train = [images[x] for x in list_partition[0]]
gts_train = [gts[x] for x in list_partition[0]]

images_val = [images[x] for x in list_partition[1]]
gts_val = [gts[x] for x in list_partition[1]]

images_test = [images[x] for x in list_partition[1]]
gts_test = [gts[x] for x in list_partition[1]]

## Verification correspondances
for i, j in zip(images_train, gts_train):
    print(i, j)
print()
for i, j in zip(images_val, gts_val):
    print(i, j)

# creation des dictionnaires d'entrainement
if determine_loss != "PD":
    train_files = [{"image": img, "label": gt} for img, gt in zip(images_train, gts_train)]
    val_files = [{"image": img, "label": gt} for img, gt in zip(images_val, gts_val)]
    test_files = [{"image": img, "label": gt} for img, gt in zip(images_test, gts_test)]
else:

    mask = sorted(glob( f"{img_file}/masked_pos_deco*.nii.gz"))
    if noise_data:
        mask += sorted(glob( f"patch_vide/pos_deco*.nii.gz"))
    print(mask)
    # creation des listes de trains ou sont situés les patchs
    masks_train = [mask[x] for x in list_partition[0]]
    masks_val = [mask[x] for x in list_partition[1]]
    masks_test = [mask[x] for x in list_partition[1]]

    train_files = [{"image": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_train, gts_train, masks_train)]
    val_files = [{"image": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_val, gts_val, masks_val)]
    test_files = [{"image": img, "label": gt, "mask": mask} for img, gt, mask in zip(images_test, gts_test, masks_test)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model unet
if residual:
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=(norm)
    ).to(device)
else:
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=0,
        norm=(norm)
    ).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr)

print(summary(model, input_size=(1,96,96,96)))
# transformees : chargemnt des images + rotation et flips aléatoires selon la seed
if determine_loss != "PD":
    train_trans = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
            RandSpatialCropd(["image", "label"], roi_size=size_patch, random_size=False),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_trans = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            RandSpatialCropd(["image", "label"], roi_size=size_patch, random_size=False),
            ToTensord(keys=["image", "label"]),
        ]
        )
    test_trans = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ]
        )
else:
    train_trans = Compose(
        [
            LoadImaged(keys=["image", "label", "mask"]),
            ScaleIntensityd(keys=["image", "label", "mask"]),
            AddChanneld(keys=["image", "label", "mask"]),
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
            AddChanneld(keys=["image", "label", "mask"]),
            RandSpatialCropd(["image", "label", "mask"], roi_size=size_patch, random_size=False),
            ToTensord(keys=["image", "label", "mask"]),
        ]
    )
    test_trans = Compose(
        [
            LoadImaged(keys=["image", "label", "mask"]),
            ScaleIntensityd(keys=["image", "label", "mask"]),
            AddChanneld(keys=["image", "label", "mask"]),
            ToTensord(keys=["image", "label", "mask"]),
        ]
    )
############################### datasets ####################################################
# création des datasets
check_ds = CacheDataset(data=train_files, transform=train_trans)
check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available(), shuffle=shuffle)

val_ds = CacheDataset(data=val_files, transform=val_trans)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available())

test_ds = CacheDataset(data=test_files, transform=test_trans)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
################################################ training###############################################################

best_metric = 100000
best_metric_epoch = -1

# boucle dentrainement
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
        if determine_loss == "D" or determine_loss == "P":
            inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            loss = loss_function1(outputs, labels)
            epoch_loss_D += loss.item()

        else:
            inputs, labels, masks = (batch_data["image"].to(device), batch_data["label"].to(device), batch_data["mask"].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss, dice_norm, dice_frag = loss_function1(outputs, labels, masks)
            epoch_loss_D += loss.item()
            epoch_loss_norm_dice += dice_norm.item()
            epoch_loss_frag += dice_frag.item()

        loss.backward()
        optimizer.step()
        # epoch_loss += loss.item()
        print(
            f"{step}/{len(check_loader)}, "
            f"train_loss: {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(root_dir, "last_model.pth"))
    print("step : ", step)
    if determine_loss =="D" or determine_loss == "P":
        epoch_loss_D /= step
        training_loss.append(epoch_loss_D)
        print(f"epoch {epoch + 1} average Dice loss: {epoch_loss_D:.4f}")
    else:
        epoch_loss_D /= step
        training_loss.append(epoch_loss_D)
        epoch_loss_norm_dice /= step
        training_loss_dice.append(epoch_loss_norm_dice)
        epoch_loss_frag /= step
        training_loss_dice_frag.append(epoch_loss_frag)
        print(f"epoch {epoch + 1} average Dice loss: {epoch_loss_D:.4f}, BCE Loss: {epoch_loss_norm_dice:.4f}, DBCE Loss: {epoch_loss_frag:.4f}")

    model.eval()
    epoch_loss_D = 0
    epoch_loss_norm_dice = 0
    epoch_loss_frag = 0
    with torch.no_grad():
        metric_count = 0
        for val_data in val_loader:
            metric_count += 1
            if determine_loss == "D" or determine_loss == "P":
                # evaluation sur les patches
                val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))
                val_outputs = model(val_inputs)
                val_outputs = torch.sigmoid(val_outputs)
                value = loss_function1(val_outputs, val_labels)
                epoch_loss_D += value.item()
            else:
                # evaluation sur les patches
                val_inputs, val_labels, val_masks = (val_data["image"].to(device), val_data["label"].to(device), val_data["mask"].to(device) )
                val_outputs = model(val_inputs)
                val_outputs = torch.sigmoid(val_outputs)
                value1, val_norm_dice, val_dice_frag = loss_function1(val_outputs, val_labels, val_masks)
                epoch_loss_D += value1.item()
                epoch_loss_norm_dice += val_norm_dice.item()
                epoch_loss_frag += val_dice_frag.item()

        if determine_loss == "D" or determine_loss == "P":
            epoch_loss_D /= metric_count
            validation_loss.append(epoch_loss_D)
            metric = epoch_loss_D
            print(f"epoch {epoch + 1} average Dice loss: {epoch_loss_D:.4f}")

        else:
            epoch_loss_D /= metric_count
            epoch_loss_norm_dice /= metric_count
            epoch_loss_frag /= metric_count

            validation_loss.append(epoch_loss_D)
            validation_loss_dice.append(epoch_loss_norm_dice)
            validation_loss_dice_frag.append(epoch_loss_frag)

            metric = epoch_loss_D
            print(f"epoch {epoch + 1} average Dice loss: {epoch_loss_D:.4f}, BCE Loss: {epoch_loss_norm_dice:.4f}, DBCE Loss: {epoch_loss_frag:.4f}")

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

if determine_loss == "D" or determine_loss == "P":
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss
    plt.plot(x, y, "-", label="Training Loss")
    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.plot(x, y, ":", label="Validation Loss")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(root_dir+"/evolution_training.png")

    torch.save((x, y), os.path.join(root_dir, f"{determine_loss}_trainingLoss.pth"))
    torch.save((x, y), os.path.join(root_dir, f"{determine_loss}_validationLoss.pth"))
    plt.close()
else:
    # dice loss
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss
    plt.plot(x, y, "-", label="D_Training Loss")
    torch.save((x, y), os.path.join(root_dir, "Dice_trainingLoss.pth"))

    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.plot(x, y, ":", label="D_Validation Loss")


    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(root_dir + "/dice_training.png")
    torch.save((x, y), os.path.join(root_dir, "Dice_validationLoss.pth"))
    torch.save((x, y), os.path.join(root_dir, "Dice_trainingLoss.pth"))
    plt.close()
    # BCE loss
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss_dice
    plt.plot(x, y, "-", label="BCE_Training Loss")
    torch.save((x, y), os.path.join(root_dir, "BCE_trainingLoss.pth"))

    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss_dice
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.plot(x, y, ":", label="BCE_Validation Loss")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(root_dir + "/BCE_training.png")
    torch.save((x, y), os.path.join(root_dir, "BCE_validationLoss.pth"))
    torch.save((x, y), os.path.join(root_dir, "BCE_trainingLoss.pth"))
    plt.close()
    # DBCE loss
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss_dice_frag
    plt.plot(x, y, "-", label="DBCE_Training Loss")

    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss_dice_frag
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.plot(x, y, ":", label="DBCE_Validation Loss")


    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(root_dir + "/DBCE_training.png")

    torch.save((x, y), os.path.join(root_dir, "DBCE_validationLoss.pth"))
    torch.save((x, y), os.path.join(root_dir, "DBCE_trainingLoss.pth"))
    plt.close()
    # BCE loss et Dice loss séparées
    x = [i + 1 for i in range(max_epochs)]
    y = training_loss
    plt.plot(x, y, "-", label="Dice Loss")

    x = [i + 1 for i in range(max_epochs)]
    y = validation_loss_dice
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.plot(x, y, ":", label="BCE Loss")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(root_dir + "/DBCE_separe_training.png")
    plt.close()
training_config = {
    "input": input_bruit,
    "output": output_bruit,
    "optimizer": "adam",
    "norm": norm,
    "patch_vide": noise_data,
    "batch_size": batch_size,
    "learning_rate": lr,
    "epochs": max_epochs,
    "loss": determine_loss,
    "patch_size": size_patch,
    "type_training": args.dataset,
    "best_epoch": best_metric_epoch,
    "best_diceloss": best_metric,
    "seed": seed,
    "residual architecture": args.residual_archi
}
# Save config in json file
with open( f'{root_dir}/config_training.json', 'w') as outfile:
    json.dump(training_config, outfile)

sw_batch_size = 5
# evaluation sur le set de validation
model.eval()
with torch.no_grad():
    for (test_data, i) in zip(test_loader, range(len(test_files))):
        val_inputs, val_labels = (test_data["image"], test_data["label"])
        val_outputs = sliding_window_inference(val_inputs.float(), size_patch, sw_batch_size, model, mode = "gaussian", overlap = 0.5)
        val_outputs = torch.sigmoid(val_outputs)
        write_nifti(data=val_inputs.detach().cpu().squeeze().numpy(), file_name=f"{root_dir}/inputs_{i}.nii.gz", resample=False)
        write_nifti(data=val_outputs.detach().cpu().squeeze().numpy(), file_name =f"{root_dir}/output_{i}.nii.gz", resample = False)
        write_nifti(data=val_labels.detach().cpu().squeeze().numpy(), file_name =f"{root_dir}/label_{i}.nii.gz", resample = False)
