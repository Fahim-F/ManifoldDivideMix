from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image
import json
import os
from datasets.Asymmetric_Noise import *
import math
from utils.mypath import Path
import csv

import pdb
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, ood_noise, root_dir, transform, mode, noise_file='/datasets/cifar100-noisy_ood_02', corruption="imagenet32", ood=[], pred=[], probability=[], log='', epoch=0,transform_st=None): 
        self.root_dir = root_dir
        self.r = r # noise ratio
        self.ood_noise = ood_noise
        self.transform = transform
        self.transform_st = transform_st
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            np.random.seed(round(math.exp(1) * 1000))
            #OOD noise
            if self.ood_noise > 0:
                self.ids_ood = [i for i, t in enumerate(train_label) if np.random.random() < self.ood_noise]
                if corruption == "imagenet32":
                    print('Corrupting CIFAR-100 with ImageNet32 data')
                    from datasets.imagenet32 import ImageNet
                    imagenet32 = ImageNet(root='/raid/home/fahimehf/Codes/Self-Semi/datasets/', size=32, train=True)
                    ood_images = imagenet32.data[np.random.permutation(np.arange(len(imagenet32)))[:len(self.ids_ood)]]
                    train_data[self.ids_ood] = ood_images
                    del imagenet32
                elif corruption == "place365":
                    print('Corrupting CIFAR-100 with Places365 data')
                    images_dir = np.array(os.listdir(Path.db_root_dir("place365")))
                    images_dir.sort()
                    ood_images = images_dir[np.random.permutation(np.arange(len(images_dir)))[:len(self.ids_ood)]]
                    ood_images = np.array([np.array(Image.open(os.path.join(Path.db_root_dir("place365"), im)).resize((32, 32), resample=2).convert('RGB')) for im in ood_images])#Better could be done
                    train_data[self.ids_ood] = ood_images

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise    
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)   
                self.ids_not_ood = [i for i in range(len(train_data)) if i not in self.ids_ood]
                noise_idx = [i for i in self.ids_not_ood if np.random.random() < (self.r/(1-self.ood_noise))]         
                
                if noise_mode == 'asym' and dataset== 'cifar100':
                    out_asym = noisify_cifar100_asymmetric(train_label, self.r)
                    noise_label = out_asym[0].tolist()
                else:
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)
                            elif noise_mode=='asym': 
                                if dataset == 'cifar10':  
                                    noiselabel = self.transition[train_label[i]]
                                    noise_label.append(noiselabel)      
                                    
                        else:    
                            noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))       
            if self.mode == 'ood':
                ood_idx = ood.nonzero()[0]   
                self.train_data = train_data[ood_idx] 
                self.noise_label = [noise_label[i] for i in ood_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))  

            if self.mode == 'all' or self.mode=='all_sup':
                self.train_data = train_data
                self.noise_label = noise_label               
            elif self.mode == 'all-ood':
                ind_idx = ood.zero()[0]   
                self.train_data = train_data[ind_idx] 
                self.noise_label = [noise_label[i] for i in ind_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))  
            else:                   
                if self.mode == "labeled":
                    data_list = []
                    pred_idx = (pred * ood).nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                                    
                elif self.mode == "unlabeled":
                    pred_idx = ((1-pred) * ood).nonzero()[0]       
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            if self.transform_st:
                img1 = self.transform(img) 
                img2 = self.transform(img) 
                img3 = self.transform_st(img) 
                img4 = self.transform_st(img) 
                return img1, img2, img3, img4, target, prob   
            else:
                img1 = self.transform(img) 
                img2 = self.transform(img) 
                return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            if self.transform_st:
                img1 = self.transform(img) 
                img2 = self.transform(img) 
                img3 = self.transform_st(img) 
                img4 = self.transform_st(img) 
                return img1, img2, img3, img4 
            else:
                img1 = self.transform(img) 
                img2 = self.transform(img) 
                return img1, img2       
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1, img2 = self.transform(img)            
            return (img1, img2), target, index
        elif self.mode=='all_sup':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)            
            return img1, target, index             
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
