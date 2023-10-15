import torch
import numpy as np
from losses.robust_losses import SCELoss
import torch.nn.functional as F

class SemiLoss(object):
    def __call__(self,  outputs_x, targets_x, outputs_u, targets_u, mask_x, mask_u, num_classes, gpu, loss_type = 'CE'):
        probs_u = torch.softmax(outputs_u, dim=1)
        # criterion_robust = NCEandRCE(num_classes=num_classes).cuda(gpu)
        criterion_robust = SCELoss(out_type='individual', num_classes=num_classes).cuda(gpu)

        if loss_type == 'RL':
            Lx = (mask_x* criterion_robust(outputs_x, targets_x )).mean()
        else:
            Lx = -torch.mean(torch.sum((mask_x * F.log_softmax(outputs_x, dim=1) * targets_x), dim=1))
        Lu = torch.mean(torch.mean((probs_u - targets_u)**2, dim=1)*mask_u) 

        return Lx, Lu

## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return float(current)
