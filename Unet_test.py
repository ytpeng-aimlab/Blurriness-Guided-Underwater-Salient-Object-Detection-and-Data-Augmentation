import os
import argparse
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim  
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import torch.nn.functional as F

from data_loader_UNet import ToTensorLab
from data_loader_UNet import SalObjDataset
from model.unet_model_ResNet50_def_decoder_caf import UNet
# from model.unet_model_ResNet50_def_decoder_add import UNet
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calculate_mae(pred, gt):
    return torch.mean(torch.abs(pred - gt))

def calculate_fm(pred, gt, beta2=0.3):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    thresholds = torch.linspace(0, 1, steps=255)
    fm_max = 0

    for threshold in thresholds:
        binary_pred = (pred >= threshold).float()
        tp = torch.sum(binary_pred * gt)
        fp = torch.sum(binary_pred * (1 - gt))
        fn = torch.sum((1 - binary_pred) * gt)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        fm = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
        fm_max = max(fm_max, fm)

    return fm_max

def calculate_f_all(pred, gt, beta2=0.3):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt   = (gt >= 0.5).float()

    thresholds = torch.linspace(0, 1, steps=256, device=pred.device)

    fm_list = []
    wfm_numer = 0.0
    wfm_denom = 0.0

    for threshold in thresholds:
        binary_pred = (pred >= threshold).float()

        tp = torch.sum(binary_pred * gt)
        fp = torch.sum(binary_pred * (1 - gt))
        fn = torch.sum((1 - binary_pred) * gt)

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        fm        = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

        fm_list.append(fm)

        pred_ones = torch.sum(binary_pred)
        wfm_numer += fm * pred_ones
        wfm_denom += pred_ones

    fm_tensor = torch.stack(fm_list)  # list 裝的是 scalar tensor，先疊成一個大的 tensor

    f_max = torch.max(fm_tensor)
    f_avg = torch.mean(fm_tensor)
    if wfm_denom > 0:
        f_w = wfm_numer / wfm_denom
    else:
        f_w = torch.tensor(0.0, device=pred.device)

    return f_max, f_avg, f_w

def calculate_em(pred, gt, kernel_size=15):
    if len(pred.shape) > 4:
        pred = pred.squeeze() 
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) > 4:
        gt = gt.squeeze()
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
        
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt >= 0.5).float()

    padding = kernel_size // 2
    local_avg = F.avg_pool2d(pred, kernel_size, stride=1, padding=padding)

    align_matrix = 2 * pred * gt + (1 - pred) * (1 - gt)

    local_term = 1 - torch.abs(pred - local_avg)

    em_score = torch.mean((align_matrix * local_term).clamp(0, 1))

    return em_score

def gradient_x(img):
    # 確保輸入維度正確
    if len(img.shape) == 4:  # [B,C,H,W]
        pass
    elif len(img.shape) == 3:  # [C,H,W]
        img = img.unsqueeze(0)
    elif len(img.shape) == 2:  # [H,W]
        img = img.unsqueeze(0).unsqueeze(0)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_x = sobel_x.view(1, 1, 3, 3).to(img.device)
    
    if img.shape[1] > 1:
        img = img[:, 0:1, :, :]
    
    return F.conv2d(img, sobel_x, padding=1)

def gradient_y(img):
    if len(img.shape) == 4:  # [B,C,H,W]
        pass
    elif len(img.shape) == 3:  # [C,H,W]
        img = img.unsqueeze(0)
    elif len(img.shape) == 2:  # [H,W]
        img = img.unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    sobel_y = sobel_y.view(1, 1, 3, 3).to(img.device)
    
    if img.shape[1] > 1:
        img = img[:, 0:1, :, :]
    
    return F.conv2d(img, sobel_y, padding=1)

def object_similarity(pred, gt):
    if torch.sum(gt) == 0:
        return torch.tensor(0.0, device=pred.device)
    
    x_pred = gradient_x(pred)
    y_pred = gradient_y(pred)
    x_gt = gradient_x(gt)
    y_gt = gradient_y(gt)
    
    sx = (2 * x_pred * x_gt + 1e-8) / (x_pred**2 + x_gt**2 + 1e-8)
    sy = (2 * y_pred * y_gt + 1e-8) / (y_pred**2 + y_gt**2 + 1e-8)

    sx = sx.squeeze()
    sy = sy.squeeze()
    
    gt_flat = gt.squeeze()
    value_sum = torch.sum(gt_flat)
    x_sum = torch.sum((sx * gt_flat))
    y_sum = torch.sum((sy * gt_flat))
    score = (x_sum + y_sum) / (value_sum * 2 + 1e-8)
    
    return score

def region_similarity(pred, gt):
    mu_pred = torch.mean(pred)
    mu_gt = torch.mean(gt)
    
    sigma_pred = torch.std(pred)
    sigma_gt = torch.std(gt)
    
    sigma_pred_gt = torch.mean((pred - mu_pred) * (gt - mu_gt))
    
    r1 = (2 * mu_pred * mu_gt + 1e-8) / (mu_pred**2 + mu_gt**2 + 1e-8)
    r2 = (2 * sigma_pred_gt + 1e-8) / (sigma_pred**2 + sigma_gt**2 + 1e-8)
    
    return r1 * r2

def calculate_sm(pred, gt, alpha=0.5):
    if len(pred.shape) > 4:
        pred = pred.squeeze() 
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) > 4:
        gt = gt.squeeze()
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt >= 0.5).float()
    
    fg = torch.mean(gt)
    o_fg = object_similarity(pred, gt)
    o_bg = object_similarity(1-pred, 1-gt)
    
    s_o = fg * o_fg + (1-fg) * o_bg
    s_r = region_similarity(pred, gt)
    sm = alpha * s_o + (1 - alpha) * s_r
    
    return sm

def MAELoss(pred, label):
    pred = pred.type(torch.FloatTensor)
    label = label.type(torch.FloatTensor)
    # normalization
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    flat_pred = pred.flatten()
    flat_label = label.flatten()
    return torch.mean(torch.abs(flat_pred - flat_label))

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi + 1e-8)
    return dn

def save_output(image_name, i, pred, d_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    img_stem = img_name.split(".")[0]
    print((d_dir + img_stem + '.png saved'))
    imo.save(d_dir + img_stem + '.png')

def get_test_dataset(sal_mode='uw'):
    img_suffix = '.jpg'
    label_suffix = '.png'
    if sal_mode == 'uw':
        image_root = './dataset/UW800/test/'
        ground_truth_root = './dataset/UW800/test_labels/'
        defocus_root = './dataset/UW800/test_defocus/'
    return image_root, ground_truth_root, defocus_root, img_suffix, label_suffix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Select which dataset to test, test through defocus or not,and whether to save the result')
    parser.add_argument('dataset', type=str, default='o',
                        choices=['e', 'p', 'd', 'h', 't', 'm_r', 'o', 'uw', 'uwv'],
                        help="Available: 'e':ECSSD, 'p':PASCALS, 'd':DUTOMRON, 'h':HKU-IS, 't':DUTS-TE, 'm_r':MSRA-B, 'o':Ours, 'uw':Underwater Images, default:'o'")
    parser.add_argument("-s", "--save", type=str, default='true',
                        choices=['false', 'true'], help='whether to save result, default:"false"')
    parser.add_argument("-d", "--defocus", type=str, default='true',
                        choices=['false', 'true'], help='whether to use defocusnet, default:"true"')
    parser.add_argument("-b", "--backbone", type=str, default='resnet50',
                        choices=['resnet34', 'resnet18', 'resnet50', 'vgg16'],
                        help='which backbone to use, default:"resnet34"')
    parser.add_argument("-p", "--pretrain", type=str, default='unet_model_ResNet50.pth',
                        help='specify pretrain model name, default:"None"')

    config = parser.parse_args()
    image_root, ground_truth_root, defocus_root, img_suffix, label_suffix = get_test_dataset(config.dataset)

    print('Testing Dataset:', image_root.split('/')[2])
    Add_defocus = True if config.defocus == 'true' else False
    print('Using defocus net:', Add_defocus)
    print('Using backbone:', config.backbone)

    image_dir = image_root
    defocus_dir = defocus_root
    prediction_dir = ground_truth_root

    result_dir = './dataset/result/newval/unet_model_ResNet50/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model_dir = './saved_models/paper/' + config.pretrain
    image_ext = img_suffix
    label_ext = label_suffix

    img_name_list = glob.glob(image_dir + '*' + '.jpg') + glob.glob(image_dir + '*' + '.png')

    def_name_list = []
    lab_name_list = []
    for img_path in img_name_list:
        img_name = img_path.split("/")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        def_name_list.append(defocus_dir + imidx + '_def.png')
        lab_name_list.append(prediction_dir + imidx + label_ext)

    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=lab_name_list,
        def_name_list=def_name_list,
        transform=ToTensorLab(flag=0, mean=[0, 0, 0], std=[1, 1, 1], fmean=0, fstd=1),
        mode='test'
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    print("---")
    print("test images: ", len(img_name_list))
    print("test labels: ", len(lab_name_list))
    print("test defocus: ", len(def_name_list))
    print("---")

    net = UNet(3, 1)
    if torch.cuda.is_available():
        net.cuda()

    # Loading Model
    training_state = torch.load(model_dir)
    net.load_state_dict(training_state['model_state'])
    print('Load model:', model_dir.split('/')[-1])
    net.eval()

    total_mae = 0.0
    total_fmax = 0.0
    total_favg = 0.0
    total_fw = 0.0
    total_em = 0.0
    total_sm = 0.0
    test_count = 0

    for i_test, data_test in enumerate(test_salobj_dataloader):
        label_test = data_test['label'].type(torch.FloatTensor)
        inputs_test = data_test['image'].type(torch.FloatTensor)
        defocus = data_test['defocus'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test, defocus, label_test = inputs_test.cuda(), defocus.cuda(), label_test.cuda()

        defocus_3ch = torch.cat((defocus, defocus, defocus), 1)
        d1, *_ = net(inputs_test, defocus_3ch)

        pred = normPRED(d1)

        mae_val = calculate_mae(pred, label_test)
        fmax_val, favg_val, fw_val = calculate_f_all(pred, label_test)
        em_val = calculate_em(pred, label_test)
        sm_val = calculate_sm(pred, label_test)

        total_mae += mae_val.item()
        total_fmax += fmax_val.item()
        total_favg += favg_val.item()
        total_fw += fw_val.item()
        total_em += em_val.item()
        total_sm += sm_val.item()

        test_count += 1
        print('  MAE: {:.4f}, Fmax: {:.4f}, Favg: {:.4f}, Fw: {:.4f}, E-measure: {:.4f}, S-measure: {:.4f}'
              .format(mae_val.item(), fmax_val.item(), favg_val.item(), fw_val.item(), em_val.item(), sm_val.item()))

        if config.save == 'true':
            save_output(img_name_list[i_test], i_test, pred, result_dir)

    print('\n---------- Evaluation Results over {} images ----------'.format(test_count))
    print('  Average MAE: {:.4f}'.format(total_mae / test_count))
    print('  Average Fmax: {:.4f}'.format(total_fmax / test_count))
    print('  Average Favg: {:.4f}'.format(total_favg / test_count))
    print('  Average Fw: {:.4f}'.format(total_fw / test_count))
    print('  Average E-measure: {:.4f}'.format(total_em / test_count))
    print('  Average S-measure: {:.4f}'.format(total_sm / test_count))
    print('------------------------------------------------------')