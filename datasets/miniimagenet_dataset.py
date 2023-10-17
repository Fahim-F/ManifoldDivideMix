from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MiniImagenet(Dataset): 
    def __init__(self, root_dir, transform, ind_ratio, color='red', mode='all', ood=[], pred=[], probability=[], log='', transform_st=None): 
        self.root = root_dir
        self.transform = transform
        self.transform_st = transform_st
        self.mode = mode
        self.r = ind_ratio # noise ratio
        num_class = 100
        self.probability = probability
        self.problem_files = []
        if self.mode=='test':
            with open(self.root+'split/clean_validation') as f:            
                lines=f.readlines()
            val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                img_path = 'validation/' +target+'/'+img
                target = int(target)
                val_imgs.append(img_path)
                self.val_labels[img_path]=target   
            self.val_imgs = np.array(val_imgs)                           
        else:    
            noise_file = '{}_noise_nl_{}'.format(color,self.r)
            with open(self.root+'split/'+noise_file) as f:
                lines=f.readlines()   
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                train_path = 'all_images/'
                train_imgs.append(train_path + img)
                self.train_labels[train_path + img]=target   
            train_imgs = np.array(train_imgs)
            if self.mode == 'all'  or self.mode=='all_sup':
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = (pred * ood).nonzero()[0]
                    self.probability = [self.probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(pred_idx)))                                  
                elif self.mode == "unlabeled":
                    pred_idx = ((1-pred) * ood).nonzero()[0]                                               
                    print("%s data has a size of %d"%(self.mode,len(pred_idx)))  
               
                self.train_imgs = train_imgs[pred_idx]
                                           
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            img = Image.open(self.root+img_path).convert('RGB')    
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
            img_path = self.train_imgs[index]
            img = Image.open(self.root+img_path).convert('RGB')    
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
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]   
            img = Image.open(self.root+img_path).convert('RGB')
            img1, img2 = self.transform(img)            
            return (img1, img2), target #, index
          
        elif self.mode=='all_sup':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')
            img1 = self.transform(img)            
            return img1, target, index    
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(img)            
            return img, target
        
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    
