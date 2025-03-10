# 350 438 269 417 419 for cat defocus and input
# 222 194 for depthPDP
# 302 402 461 for original BASNet(one input)
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import argparse
import time
import os
import csv
import json
from functools import partial
import numpy as np
import glob
import matplotlib.pyplot as plt
from data_loader_UNet import Rescale, RescaleT, RandomCrop, CenterCrop, ToTensor, ToTensorLab, SalObjDataset, _collate_fn
# from model.BASNet_mix import BASNet
from model.unet_model_ResNet50_def_decoder_add import UNet
#from model.unet_model_ResNet50_def_decoder_caf import UNet
import os

import pytorch_ssim
import pytorch_iou

starttime = time.time()


# class Consine_Loss(nn.Module):
#     def __init__(self, alpha_1=1):
#         super(Consine_Loss, self).__init__()
#         self.cosineL = nn.CosineSimilarity(dim=1)
#         self.weight = alpha_1

#     def forward(self, pred, Label):
#         # Foreground
#         pred_patch = pred.unfold(2, 4, 4).unfold(
#             3, 4, 4).flatten(0, 3).flatten(-2, -1)
#         Label_patch = Label.unfold(2, 4, 4).unfold(
#             3, 4, 4).flatten(0, 3).flatten(-2, -1)

#         cLoss = 1 - torch.abs(self.cosineL(pred_patch, Label_patch).mean())
#         return cLoss * self.weight


def get_test_dataset(sal_mode='uw'):
    img_suffix = '.jpg'
    label_suffix = '.png'
    if sal_mode == 'uw':  # 400 imgs
        image_root = './dataset/UW800/train/'
        ground_truth_root = './dataset/UW800/train_labels/'
        # './dataset/New_Underwater/new/train_defocus/'
        defocus_root = './dataset/UW800/train_defocus/'

    return image_root, ground_truth_root, defocus_root, img_suffix, label_suffix

def validate_model(net, dataloader):
    net.eval()
    val_ite_num = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            val_ite_num += 1
            inputs, defocus, labels = data['image'], data['defocus'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            defocus = defocus.type(torch.FloatTensor)
            if torch.cuda.is_available():
                labels = labels.cuda()
                inputs = inputs.cuda()
                defocus = defocus.cuda()

            defocus = torch.cat((defocus, defocus, defocus), 1)
            d0, d1, d2, d3, d4, d5 = net(inputs, defocus)

            maeloss = MAELoss(d0, labels)
            val_running_loss += maeloss.item()
            
    return val_running_loss / val_ite_num, val_ite_num

# Create a function to plot learning curves
def plot_learning_curves(training_stats, save_dir):
    """Plot and save learning curves from training statistics"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.plot(
        [stat['epoch'] for stat in training_stats],
        [stat['train_loss'] for stat in training_stats],
        label='Training Loss'
    )
    
    val_stats = [stat for stat in training_stats if 'val_loss' in stat]
    if val_stats:
        plt.plot(
            [stat['epoch'] for stat in val_stats],
            [stat['val_loss'] for stat in val_stats],
            label='Validation Loss'
        )
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300)
    
    # Plot MAE
    plt.figure(figsize=(12, 8))
    plt.plot(
        [stat['epoch'] for stat in training_stats],
        [stat['mae_loss'] for stat in training_stats],
        label='Training MAE'
    )
    
    if val_stats:
        plt.plot(
            [stat['epoch'] for stat in val_stats],
            [stat['val_mae'] for stat in val_stats],
            label='Validation MAE'
        )
    
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/mae_curve.png", dpi=300)
    
    # Plot component losses (BCE, SSIM, IOU)
    plt.figure(figsize=(12, 8))
    plt.plot(
        [stat['epoch'] for stat in training_stats],
        [stat['bce_loss'] for stat in training_stats],
        label='BCE Loss'
    )
    plt.plot(
        [stat['epoch'] for stat in training_stats],
        [stat['ssim_loss'] for stat in training_stats],
        label='SSIM Loss'
    )
    plt.plot(
        [stat['epoch'] for stat in training_stats],
        [stat['iou_loss'] for stat in training_stats],
        label='IOU Loss'
    )
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component Values')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/component_losses.png", dpi=300)
    
    plt.close('all')

# --------0. get parser info-------------
parser = argparse.ArgumentParser(
    description='Select which dataset to train and whether to use defocus net')
parser.add_argument('dataset', type=str, default='o', choices=['e', 'p', 'd', 'h', 't', 'm_r', 'o', 'uw', 'uwb'],
                    help="Available: 'e':ECSSD \n 'p':PASCALS \n 'd':DUTOMRON \n 'h':HKU-IS \n 't':DUTS-TE \n 'm_r':MSRA-B \n 'o':Ours \n 'uw':Underwater Images,default:'o'")
parser.add_argument("-d", "--defocus", type=str, default='true',
                    choices=['false', 'true'], help='whether to use defocusnet, default:"true"')
parser.add_argument("-f", "--freeze", type=str, default='',
                    choices=['basnet', ''], help='which network to freeze, default:"None"')
parser.add_argument("-p", "--pretrain", type=str, default='',
                    help='specify pretrain model name, default:"None"')
parser.add_argument("-b", "--backbone", type=str, default='resnet34', choices=[
                    'resnet34', 'resnet18', 'resnet50', 'vgg16'], help='which backbone to use, default:"resnet34"')
parser.add_argument("-val_freq", type=int, default=50,
                   help='validation frequency (epochs), default:10')
parser.add_argument("-save_dir", type=str, default='./training_stats',
                   help='directory to save training statistics and plots')
config = parser.parse_args()

# Create directories for saving statistics and plots
stats_dir = os.path.join(config.save_dir, f"{config.dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(stats_dir, exist_ok=True)

image_root, ground_truth_root, defocus_root, img_suffix, label_suffix = get_test_dataset(
    config.dataset)

exp_name = "ADDF_unet_model_ResNet50_"
model_dir = "./saved_models/paper/"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# xxx.pth. leave it as an empty string if there is no needed to load.
pretrained_weight_filename = config.pretrain
batch_size_train = 8 #8 no size 256
batch_size_val = 16
epoch_num = 751
lr = 0.00005

print('\n')
print('Batch size:', batch_size_train)
print('Training Dataset:', image_root.split('/')[2])
Add_defocus = True if config.defocus == 'true' else False
print('Using defocus net:', Add_defocus)
print('Using backbone:', config.backbone)
print('\n')
opt = {'dataset': image_root.split('/')[2], 'backbone': config.backbone, 'defocus': Add_defocus,
       'pretrained_weight': pretrained_weight_filename.split('.')[0]}

# Save training configuration
with open(f"{stats_dir}/config.json", 'w') as f:
    config_dict = {
        'dataset': config.dataset,
        'defocus': config.defocus,
        'freeze': config.freeze,
        'pretrain': config.pretrain,
        'backbone': config.backbone,
        'val_freq': config.val_freq,
        'batch_size_train': batch_size_train,
        'batch_size_val': batch_size_val,
        'epoch_num': epoch_num,
        'learning_rate': lr,
        'exp_name': exp_name,
        'train_size': 0,  # Will be updated later
        'val_size': 0     # Will be updated later
    }
    json.dump(config_dict, f, indent=4)

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
# consine_Loss = Consine_Loss()

def MAELoss(pred, label):
    '''self defined MAELoss'''
    pred = pred.type(torch.FloatTensor)
    label = label.type(torch.FloatTensor)
    # normalization
    pred = (pred - pred.min()) / (pred.max()-pred.min())  # ct_0520
    # compute mae and accuracy
    flat_pred = pred.flatten()
    #pat_pred = (flat_pred>=2*flat_pred.mean()).type(torch.bool).type(torch.FloatTensor)
    flat_label = label.flatten()
    return torch.mean(torch.abs(flat_pred-flat_label))


def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    # consine_out = consine_Loss(pred,target)

    loss = bce_out + ssim_out + iou_out  # + consine_out

    return loss, bce_out, ssim_out,  iou_out  # , consine_out


# ,defocus,enhenced_defocus):
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, labels_v):

    loss0, bce_out0, ssim_out0, iou_out0 = bce_ssim_loss(d0, labels_v)
    loss1, bce_out1, ssim_out1, iou_out1 = bce_ssim_loss(d1, labels_v)
    loss2, bce_out2, ssim_out2, iou_out2 = bce_ssim_loss(d2, labels_v)
    loss3, bce_out3, ssim_out3, iou_out3 = bce_ssim_loss(d3, labels_v)
    loss4, bce_out4, ssim_out4, iou_out4 = bce_ssim_loss(d4, labels_v)
    loss5, bce_out5, ssim_out5, iou_out5 = bce_ssim_loss(d5, labels_v)
    # loss6, bce_out6, ssim_out6, iou_out6 = bce_ssim_loss(d6, labels_v)
    # loss7, bce_out7, ssim_out7, iou_out7 = bce_ssim_loss(d7, labels_v)
    #ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
#     loss = loss0[0] + loss1[0] + loss2[0] + loss3[0] + loss4[0] + loss5[0] + loss6[0] + loss7[0]#+ 5.0*lossa
    bce_tloss = bce_out0 + bce_out1 + bce_out2 + \
        bce_out3 + bce_out4 + bce_out5
    ssim_tloss = ssim_out0 + ssim_out1 + ssim_out2 + \
        ssim_out3 + ssim_out4 + ssim_out5
    iou_tloss = iou_out0 + iou_out1 + iou_out2 +\
        iou_out3 + iou_out4 + iou_out5
    #consine_tlos = consine_out0 + consine_out1 + consine_out2 + consine_out3 + consine_out4 + consine_out5 + consine_out6 + consine_out7
    # c_loss = contra_Loss(defocus,enhenced_defocus,labels_v)
    loss = bce_tloss + ssim_tloss + iou_tloss  # +consine_tlos #+ c_loss
    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f" % (loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()))
    loss_values = {
        'loss0': loss0.item(),
        'bce_tloss': bce_tloss.item(),
        'ssim_tloss': ssim_tloss.item(),
        'iou_tloss': iou_tloss.item(),
    }
    
    del loss1, loss2, loss3, loss4, loss5 , bce_out0, bce_out1, bce_out2, bce_out3, bce_out4, bce_out5, ssim_out0, ssim_out1, ssim_out2, ssim_out3, ssim_out4, ssim_out5, iou_out0, iou_out1, iou_out2, iou_out3, iou_out4, iou_out5
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

    return loss0, loss, bce_tloss, ssim_tloss, iou_tloss, loss_values  # , consine_tlos#, c_loss


# ------- 2. set the directory of training dataset --------

# 'C:\\Users\\Admin\\Documents\\Projects\\BASNet-master\\dataset\\Ours\\train\\'
tra_image_dir = image_root
# 'C:\\Users\\Admin\\Documents\\Projects\\BASNet-master\\dataset\\Ours\\train_labels\\'
tra_label_dir = ground_truth_root
# 'C:\\Users\\Admin\\Documents\\Projects\\BASNet-master\\dataset\\Ours\\train_defocus\\'
tra_defocus_dir = defocus_root
image_ext = img_suffix
label_ext = label_suffix

tra_img_name_list = glob.glob(tra_image_dir + '*' + '.jpg') + glob.glob(tra_image_dir + '*' + '.png')
tra_lbl_name_list = []
tra_def_name_list = []

for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]
    # img_name = img_path.split("\\")[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)
    tra_def_name_list.append(tra_defocus_dir + imidx + '_def' + label_ext)

train_num = len(tra_img_name_list)


# ------- 2.1 set validation dataloader --------

val_image_dir = './dataset/UW800/test/' 
val_label_dir = './dataset/UW800/test_labels/'
val_defocus_dir = './dataset/UW800/test_defocus/'
image_ext = '.jpg'
label_ext = '.png'

val_img_name_list =  glob.glob(val_image_dir + '*' + '.jpg') + glob.glob(val_image_dir + '*' + '.png')
val_num = len(val_img_name_list)
val_lbl_name_list = []
val_def_name_list = []
for val_img_path in val_img_name_list:
    val_img_name = val_img_path.split("/")[-1]
    # val_img_name = val_img_path.split("\\")[-1]
    aaaa = val_img_name.split(".")
    bbbb = aaaa[0:-1]
    val_imidx = bbbb[0]
    for i in range(1, len(bbbb)):
        val_imidx = val_imidx + "." + bbbb[i]

    val_lbl_name_list.append(val_label_dir + val_imidx + label_ext)
    val_def_name_list.append(val_defocus_dir + val_imidx + '_def' + label_ext)

# Update train and val sizes in config
config_dict['train_size'] = train_num
config_dict['val_size'] = val_num
with open(f"{stats_dir}/config.json", 'w') as f:
    json.dump(config_dict, f, indent=4)

# if not Add_defocus:#################
    # tra_def_name_list =[]
    # val_def_name_list =[]

print("---")
print("train images: ", train_num)
print("train labels: ", len(tra_lbl_name_list))
print("train defocus: ", len(tra_def_name_list))
print("---")
print("---")
print("validation images: ", val_num)
print("validation labels: ", len(val_lbl_name_list))
print("validation defocus: ", len(val_def_name_list))
print("---")

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    def_name_list=tra_def_name_list,
    transform=ToTensorLab(flag=0, mean=[0, 0, 0], std=[1, 1, 1], fmean=0, fstd=1))
salobj_dataloader = DataLoader(
    salobj_dataset, batch_size=batch_size_train, shuffle=True) #,collate_fn=collate_fn)

# resclse and crop are done in data_loader_new.py
val_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    def_name_list=val_def_name_list,
    transform=ToTensorLab(flag=0, mean=[0, 0, 0], std=[1, 1, 1], fmean=0, fstd=1),
    mode='test')
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size_val, shuffle=False)

# ------- 3. define model --------
start_epo = 0
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
val_running_loss = 0.0
val_ite_num = 0  

# Initialize stats tracking
training_stats = []

# Create CSV file for training stats
csv_file = f"{stats_dir}/training_stats.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 'train_loss', 'train_tar_loss', 'bce_loss', 
        'ssim_loss', 'iou_loss', 'mae_loss', 'val_loss', 'val_mae',
        'learning_rate', 'time_elapsed'
    ])

# net = BASNet(4, 1)
net = UNet(3, 1)
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")

optimizer = optim.Adam(net.parameters(), lr=lr, betas=(
    0.9, 0.999), eps=1e-08, weight_decay=0)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, 
    verbose=True, threshold=0.001, threshold_mode='rel'
)

# ------- load state_dict --------
#net.load_state_dict(torch.load(model_dir + pretrained_weight_filename))
# Comment below
if pretrained_weight_filename:
    if pretrained_weight_filename == 'basnet.pth':

        old_weight = net.state_dict()  # load pretrained weights of model
        #new_weight =  old_weight.update(torch.load(model_dir + pretrained_weight_filename))
        new_weight = {k: i for (k, i) in torch.load(model_dir + pretrained_weight_filename).items(
        ) if old_weight[k].size() == torch.load(model_dir + pretrained_weight_filename)[k].size()}
        old_weight.update(new_weight)

        #print('=================weight {} not load================='.format(old_weight.keys() - new_weight.keys()))
        net.load_state_dict(old_weight)
    else:

        print('\n Load model:', pretrained_weight_filename, '\n')
        training_state = torch.load(
            model_dir + pretrained_weight_filename, map_location=torch.device('cpu'))
        start_epo = training_state['epoch']+1
        ite_num = training_state['iter']

        # new_weight = net.state_dict() #load pretrained weights of model
        # new_weight.update(training_state['model_state'])
        # net.load_state_dict(torch.load('/home/ytpeng0418/YCLin/BASNet/saved_models/paper/unet_model_ResNet50_def_encoder_hybridloss_UW800_best_650.pth',map_location=torch.device('cpu'))['model_state'],strict=False)
        net.load_state_dict(training_state['model_state'],strict=False)

        # if len(training_state['optimizer_state']['param_groups'][0]['params']) == len(optimizer.state_dict()):
        # print('\n Load Optimizer \n')
        # new_optimizer = optimizer.state_dict()
        # new_optimizer.update(training_state['optimizer_state'])
        # optimizer.load_state_dict(new_optimizer)
        # if 'scheduler_state' in training_state and training_state['scheduler_state'] is not None:
        #     scheduler.load_state_dict(training_state['scheduler_state'])

# ------- 5. training process --------
print("---start training...")
miniMAE = 0.05
writer = SummaryWriter(log_dir=f'{stats_dir}/tensorboard')

for epoch in range(start_epo, epoch_num):
    epoch_start_time = time.time()
    net.train()
    epoch_loss = 0.0
    epoch_tar_loss = 0.0
    epoch_bce_loss = 0.0
    epoch_ssim_loss = 0.0
    epoch_iou_loss = 0.0
    epoch_mae_loss = 0.0
    batch_count = 0

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1
        batch_count += 1

        inputs, defocus, labels = data['image'], data['defocus'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        defocus = defocus.type(torch.FloatTensor)
        if torch.cuda.is_available():
            labels = labels.cuda()
            inputs = inputs.cuda()
            defocus = defocus.cuda()

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        defocus = torch.cat((defocus, defocus, defocus),1)
        d0, d1, d2, d3, d4, d5 = net(inputs, defocus)

        # Calculate losses
        loss2, loss, bce_tloss, ssim_tloss, iou_tloss, loss_values = muti_bce_loss_fusion(
            d0, d1, d2, d3, d4, d5, labels)
        maeloss = MAELoss(d0, labels)
        
        # Accumulate epoch losses
        epoch_loss += loss.item()
        epoch_tar_loss += loss2.item()
        epoch_bce_loss += bce_tloss.item()
        epoch_ssim_loss += ssim_tloss.item()
        epoch_iou_loss += iou_tloss.item()
        epoch_mae_loss += maeloss.item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

    # Calculate average losses for the epoch
    avg_loss = epoch_loss / batch_count
    avg_tar_loss = epoch_tar_loss / batch_count
    avg_bce_loss = epoch_bce_loss / batch_count
    avg_ssim_loss = epoch_ssim_loss / batch_count
    avg_iou_loss = epoch_iou_loss / batch_count
    avg_mae_loss = epoch_mae_loss / batch_count
    
    # Print epoch statistics
    print(f"[Epoch: {epoch+1}/{epoch_num}] "
          f"train loss: {avg_loss:.6f}, "
          f"tar loss: {avg_tar_loss:.6f}, "
          f"MAE: {avg_mae_loss:.6f}, "
          f"Time: {time.time() - epoch_start_time:.2f}s")

    # Record TensorBoard metrics for every epoch
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/MAE', avg_mae_loss, epoch)
    writer.add_scalar('Train/BCE', avg_bce_loss, epoch)
    writer.add_scalar('Train/SSIM', avg_ssim_loss, epoch)
    writer.add_scalar('Train/IOU', avg_iou_loss, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    
    # Initialize validation metrics
    val_loss = None
    val_mae = None
    
    # Run validation at specified frequency
    if epoch % config.val_freq == 0:
        val_mae, val_ite_num = validate_model(net, val_dataloader)
        val_loss = val_mae  # Since we're using MAE as the primary validation metric
        
        # Log validation metrics
        writer.add_scalar('Validation/MAE', val_mae, epoch)
        
        # Add to TensorBoard comparisons
        writer.add_scalars('Compare/MAE', {
            'Train': avg_mae_loss,
            'Validation': val_mae
        }, epoch)
        
        # Print validation results
        print(f"[Validation Epoch: {epoch+1}/{epoch_num}] Val MAE: {val_mae:.6f}")
        
        # Update learning rate based on validation performance
        scheduler.step(val_mae)
        
        # Save model if it's the best so far
        if val_mae < miniMAE:
            miniMAE = val_mae
            # Create directory for model if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            training_state = {
                'epoch': epoch,
                'iter': ite_num,
                'model_state': net.state_dict(),
                'options': opt,
                'scheduler_state': scheduler.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            model_path = f"{model_dir}{exp_name}_epo_{epoch}_mae_{val_mae:.5f}.pth"
            torch.save(training_state, model_path)
            print(f"Best model saved: {model_path}")
            
            # Also save a copy in the stats directory
            best_model_path = f"{stats_dir}/best_model.pth"
            torch.save(training_state, best_model_path)
    
    # Save Training Stats
    epoch_stats = {
        'epoch': epoch,
        'train_loss': avg_loss,
        'train_tar_loss': avg_tar_loss,
        'bce_loss': avg_bce_loss,
        'ssim_loss': avg_ssim_loss,
        'iou_loss': avg_iou_loss,
        'mae_loss': avg_mae_loss,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'lr': optimizer.param_groups[0]['lr'],
        'time_elapsed': time.time() - epoch_start_time
    }
    training_stats.append(epoch_stats)
    
    with open(csv_file, 'a', newline='') as f:
        #writer.writerow = csv.writer(f)
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            epoch, avg_loss, avg_tar_loss, avg_bce_loss,
            avg_ssim_loss, avg_iou_loss, avg_mae_loss,
            val_loss if val_loss is not None else '',
            val_mae if val_mae is not None else '',
            optimizer.param_groups[0]['lr'],
            time.time() - epoch_start_time
        ])
    
    # Plot and save learning curves periodically
    if epoch % 10 == 0 or epoch == epoch_num - 1:
        plot_learning_curves(training_stats, stats_dir)
        
        # Save training stats as JSON
        with open(f"{stats_dir}/training_stats.json", 'w') as f:
            json.dump(training_stats, f, indent=4)
    
    # Clean up for next epoch
    del d0, d1, d2, d3, d4, d5, loss2, loss, bce_tloss, ssim_tloss, iou_tloss, maeloss, inputs, defocus, labels
    torch.cuda.empty_cache()
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

# Final plots and stats
plot_learning_curves(training_stats, stats_dir)

# Final save of training stats
with open(f"{stats_dir}/training_stats_final.json", 'w') as f:
    json.dump(training_stats, f, indent=4)

writer.close()
print(f'-------------{exp_name} Done-------------')
print('Execution time: {}'.format(time.strftime(
    "%H:%M:%S", time.gmtime(time.time() - starttime))))
print(f'Training stats saved to: {stats_dir}')