from torch import nn
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt



def seperate_lul_sets(args, val_loader, model, epoch=5):

    criterion_loss = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
    model.eval()

    sample_loss= []
    with torch.no_grad():
        for images, target, _ in val_loader:
            _,_, output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion_loss(output, target.cuda(gpu, non_blocking=True))
            sample_loss.append(loss.cpu().numpy())
                    
    sample_loss = np.concatenate(sample_loss)
    sample_loss = (sample_loss-sample_loss.min())/(sample_loss.max()-sample_loss.min())    

    input_loss = sample_loss.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=20,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]     
    pred = (prob > args.p_threshold)  
    # f, ax = plt.subplots(1, 1, figsize=(12, 5))      
    # ax.hist(sample_loss[np.where(prob > args.p_threshold)], bins=1000, color='green', alpha=0.5, ec='green', label='Clean Data')
    # ax.hist(sample_loss[np.where(prob <= args.p_threshold)], bins=1000, color='blue', alpha=0.5, ec='blue', label='Noisy Data')
    # ax.legend(loc='upper right')
    # name_losscluster = 'gmmloss_' + str(epoch) + '.png'
    # plt.savefig(args.exp_dir / name_losscluster)
    return pred, prob
