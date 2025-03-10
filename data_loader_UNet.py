
# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torch.nn.functional import interpolate
#==========================dataset load==========================

class RescaleT(object):

    def __init__(self,output_size,defocus_rescale=True):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
        self.defocus_rescale = defocus_rescale
    def __call__(self,sample):
        defocus, image, label = sample['defocus'],sample['image'],sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
        lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)
        if self.defocus_rescale:
            defocus = transform.resize(defocus,(self.output_size,self.output_size),mode='constant') ##

        return {'image':img,'label':lbl,'defocus':defocus}

class Rescale(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        defocus, image, label = sample['defocus'],sample['image'],sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image,(new_h,new_w),mode='constant')
        defocus = transform.resize(defocus,(new_h,new_w),mode='constant')
        lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        return {'image':img,'label':lbl,'defocus':defocus}

class CenterCrop(object):

    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self,sample):
        defocus, image, label = sample['defocus'],sample['image'],sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
        assert((h >= new_h) and (w >= new_w))

        h_offset = int(math.floor((h - new_h)/2))
        w_offset = int(math.floor((w - new_w)/2))

        image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        defocus = defocus[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

        return {'image':image,'label':label,'defocus':defocus}

class RandomCrop(object):

    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self,sample):
        defocus, image, label = sample['defocus'],sample['image'],sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        defocus = defocus[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'image':image,'label':label,'defocus':defocus}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        defocus, image, label = sample['defocus'],sample['image'],sample['label']

        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        tmpdef = np.zeros((defocus.shape[0],defocus.shape[1],3))##

        tmpLbl = np.zeros(label.shape)

        image = image/np.max(image)
        defocus = defocus/np.max(defocus)
        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)

        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        if defocus.shape[2]==1:
            tmpdef[:,:,0] = (defocus[:,:,0]-0.485)/0.229 ##
            tmpdef[:,:,1] = (defocus[:,:,0]-0.485)/0.229 ##
            tmpdef[:,:,2] = (defocus[:,:,0]-0.485)/0.229 ##
        else:
            tmpdef[:,:,0] = (defocus[:,:,0]-0.485)/0.229 ##
            tmpdef[:,:,1] = (defocus[:,:,1]-0.456)/0.224 ##
            tmpdef[:,:,2] = (defocus[:,:,2]-0.406)/0.225 ##





        tmpLbl[:,:,0] = label[:,:,0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpdef = tmpdef.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg),
            'defocus': torch.from_numpy(tmpdef),
            'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0,mean=[0,0,0],std=[1,1,1],fmean=0,fstd=1):
        self.flag = flag
        self.mean = mean
        self.std = std
        self.fmean = fmean
        self.fstd = fstd
    def __call__(self, sample):

        defocus, image, label = sample['defocus'],sample['image'],sample['label'] ##

        tmpLbl = np.zeros(label.shape)

        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)

        # change the color space
        if self.flag == 2: # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0],image.shape[1],6))
            tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
            tmpdef = np.zeros((defocus.shape[0],defocus.shape[1],6)) ##
            tmpdeft = np.zeros((defocus.shape[0],defocus.shape[1],3)) ##
            if image.shape[2]==1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
                tmpdeft[:,:,0] = defocus[:,:,0] ##
                tmpdeft[:,:,1] = defocus[:,:,0] ##
                tmpdeft[:,:,2] = defocus[:,:,0] ##
            else:
                tmpImgt = image
                tmpdeft = defocus
            tmpImgtl = color.rgb2lab(tmpImgt)
            tmpdeft1 = color.rgb2lab(tmpdeft) ##


            # nomalize image to range [0,1]
            tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

            tmpdef[:,:,0] = (tmpdeft[:,:,0]-np.min(tmpdeft[:,:,0]))/(np.max(tmpdeft[:,:,0])-np.min(tmpdeft[:,:,0])) ##
            tmpdef[:,:,1] = (tmpdeft[:,:,1]-np.min(tmpdeft[:,:,1]))/(np.max(tmpdeft[:,:,1])-np.min(tmpdeft[:,:,1])) ##
            tmpdef[:,:,2] = (tmpdeft[:,:,2]-np.min(tmpdeft[:,:,2]))/(np.max(tmpdeft[:,:,2])-np.min(tmpdeft[:,:,2])) ##
            tmpdef[:,:,3] = (tmpdeft1[:,:,0]-np.min(tmpdeft1[:,:,0]))/(np.max(tmpdeft1[:,:,0])-np.min(tmpdeft1[:,:,0])) ##
            tmpdef[:,:,4] = (tmpdeft1[:,:,1]-np.min(tmpdeft1[:,:,1]))/(np.max(tmpdeft1[:,:,1])-np.min(tmpdeft1[:,:,1])) ##
            tmpdef[:,:,5] = (tmpdeft1[:,:,2]-np.min(tmpdeft1[:,:,2]))/(np.max(tmpdeft1[:,:,2])-np.min(tmpdeft1[:,:,2])) ##

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

            tmpdef[:,:,0] = (tmpdef[:,:,0]-np.mean(tmpdef[:,:,0]))/np.std(tmpdef[:,:,0]) ##
            tmpdef[:,:,1] = (tmpdef[:,:,1]-np.mean(tmpdef[:,:,1]))/np.std(tmpdef[:,:,1]) ##
            tmpdef[:,:,2] = (tmpdef[:,:,2]-np.mean(tmpdef[:,:,2]))/np.std(tmpdef[:,:,2]) ##
            tmpdef[:,:,3] = (tmpdef[:,:,3]-np.mean(tmpdef[:,:,3]))/np.std(tmpdef[:,:,3]) ##
            tmpdef[:,:,4] = (tmpdef[:,:,4]-np.mean(tmpdef[:,:,4]))/np.std(tmpdef[:,:,4]) ##
            tmpdef[:,:,5] = (tmpdef[:,:,5]-np.mean(tmpdef[:,:,5]))/np.std(tmpdef[:,:,5]) ##

        elif self.flag == 1: #with Lab color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            tmpdef = np.zeros((defocus.shape[0],defocus.shape[1],3))

            if image.shape[2]==1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
                tmpdef[:,:,0] = defocus[:,:,0] ##
                tmpdef[:,:,1] = defocus[:,:,0] ##
                tmpdef[:,:,2] = defocus[:,:,0] ##
            else:
                tmpImg = image
                tmpdef = defocus

            tmpImg = color.rgb2lab(tmpImg)
            tmpdef = color.rgb2lab(tmpdef)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

            tmpdef[:,:,0] = (tmpdef[:,:,0]-np.min(tmpdef[:,:,0]))/(np.max(tmpdef[:,:,0])-np.min(tmpdef[:,:,0])) ##
            tmpdef[:,:,1] = (tmpdef[:,:,1]-np.min(tmpdef[:,:,1]))/(np.max(tmpdef[:,:,1])-np.min(tmpdef[:,:,1])) ##
            tmpdef[:,:,2] = (tmpdef[:,:,2]-np.min(tmpdef[:,:,2]))/(np.max(tmpdef[:,:,2])-np.min(tmpdef[:,:,2])) ##

            tmpdef[:,:,0] = (tmpdef[:,:,0]-np.mean(tmpdef[:,:,0]))/np.std(tmpdef[:,:,0]) ##
            tmpdef[:,:,1] = (tmpdef[:,:,1]-np.mean(tmpdef[:,:,1]))/np.std(tmpdef[:,:,1]) ##
            tmpdef[:,:,2] = (tmpdef[:,:,2]-np.mean(tmpdef[:,:,2]))/np.std(tmpdef[:,:,2]) ##

        else: # with rgb color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            tmpdef = np.zeros((defocus.shape[0],defocus.shape[1],1)) ##
            image = image/np.max(image)
            defocus = defocus/np.max(defocus)
            # if image.shape[2]==1:
                # tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                # tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                # tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
            # else:
                # tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                # tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                # tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

            r_mean, g_mean, b_mean = self.mean
            r_std, g_std, b_std =self.std
            max_mean = max(self.mean)
            max_std = max(self.std)
            if image.shape[2]==1:
                tmpImg[:,:,0] = (image[:,:,0]-max_mean)/max_std
                tmpImg[:,:,1] = (image[:,:,0]-max_mean)/max_std
                tmpImg[:,:,2] = (image[:,:,0]-max_mean)/max_std
            else:
                tmpImg[:,:,0] = (image[:,:,0]-r_mean)/r_std
                tmpImg[:,:,1] = (image[:,:,1]-g_mean)/g_std
                tmpImg[:,:,2] = (image[:,:,2]-b_mean)/b_std
            if defocus.shape[2]==1:
                tmpdef[:,:,0] = (defocus[:,:,0]-self.fmean)/self.fstd
            else:
                raise ValueError("defocus map is not one channel")



        tmpLbl[:,:,0] = label[:,:,0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpdef = tmpdef.transpose((2, 0, 1)) ##
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg),
        'defocus': torch.from_numpy(tmpdef),
            'label': torch.from_numpy(tmpLbl)}##

class SalObjDataset(Dataset):
    def __init__(self,img_name_list,lbl_name_list,def_name_list,transform=None,mode='train'):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.def_name_list = def_name_list
        self.transform = transform
        self.mode = mode
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = io.imread(self.image_name_list[idx])


        if(0==len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:

            label_3 = io.imread(self.label_name_list[idx])
##
        if(0==len(self.def_name_list)):
            def_3 = np.zeros(image.shape)
        else:
            def_3 = io.imread(self.def_name_list[idx],as_gray=True)
##
        #print("len of label3")
        # print(label_3.shape)
        # print(def_3.shape)
        # print(image.shape)

        label = np.zeros(label_3.shape[0:2])
        if(3==len(label_3.shape)):
            label = label_3[:,:,0]
        elif(2==len(label_3.shape)):
            label = label_3
#
        defocus = np.zeros(def_3.shape[0:2])
        if(3==len(def_3.shape)):
            defocus = def_3[:,:,0]
        elif(2==len(def_3.shape)):
            defocus = def_3
#
        if(3==len(image.shape) and 2==len(label.shape)):
            label = label[:,:,np.newaxis]
        elif(2==len(image.shape) and 2==len(label.shape)):
            image = image[:,:,np.newaxis]
            label = label[:,:,np.newaxis]
#
        if(3==len(image.shape) and 2==len(defocus.shape)):
            defocus = defocus[:,:,np.newaxis]
        elif(2==len(image.shape) and 2==len(defocus.shape)):
            image = image[:,:,np.newaxis]
            defocus = defocus[:,:,np.newaxis]
#
        if self.mode == 'train':
        #     #vertical flipping
        #     # fliph = np.random.randn(1)
            hflip = np.random.randn(1)

            if hflip>0:
                image = image[:,::-1,:]
                label = label[:,::-1,:]
                defocus = defocus[:,::-1,:]

            # rotateFlag = np.random.randn(1)

            # if rotateFlag > 0:
            #     angle = np.int(np.random.randint(180))
            #     image = transform.rotate(image,angle,resize=False)
            #     label = transform.rotate(label,angle,resize=False)
            #     defocus = transform.rotate(defocus,angle,resize=False)

            # permuteFlag = np.random.randn(1)

            # if permuteFlag >0:
            #     permute = np.random.permutation(3)
            #     image = image[:,:,permute]

        sample = {'image':image, 'label':label, 'defocus':defocus}
        if self.transform:
            # sample = self.transform(sample)
            sample = RescaleT(256,True)(sample)
            if self.mode == 'train':
                sample = RandomCrop(224)(sample)
            sample = self.transform(sample)
        return sample


def _collate_fn(batch, size_list):
    size = np.random.choice(size_list)
    # img, defocus, mask = [list(item) for item in zip(*batch)]
    # inputs, defocus,  labels = batch['image'], batch['defocus'], batch['label']
    inputs = [item['image'] for item in batch]
    defocus = [item['defocus'] for item in batch]
    labels = [item['label'] for item in batch]

    inputs = torch.stack(inputs, dim=0)
    inputs = interpolate(inputs, size=(size, size), mode="bilinear", align_corners=False)
    defocus = torch.stack(defocus, dim=0)
    defocus = interpolate(defocus, size=(size, size), mode="bilinear", align_corners=False)
    labels = torch.stack(labels, dim=0)
    labels = interpolate(labels, size=(size, size), mode="nearest")
    sample = {'image':inputs, 'label':labels, 'defocus':defocus}
    return sample
