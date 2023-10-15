import torch
import numpy as np
from matplotlib import pyplot as plt
from cleanlab.outlier import OutOfDistribution
from cleanlab.rank import find_top_issues



def find_outliers(val_loader, model, gpu, ratio_removal):
    
    model.eval()
    embedding = []
    with torch.no_grad():
        for images, target, _ in val_loader:
            feat,_, _ = model(images.cuda(gpu, non_blocking=True))
            
            embedding.append(feat.cpu().numpy())
                    
    embedding = np.concatenate(embedding)
    out_detect = np.zeros(embedding.shape[0])

    ood = OutOfDistribution()

    train_ood_features_scores = ood.fit_score(features=embedding)
    top_train_ood_features_idxs = find_top_issues(quality_scores=train_ood_features_scores, top=ratio_removal)

    out_detect[top_train_ood_features_idxs] =1

    return torch.from_numpy(out_detect).cuda()
        