from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib


from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import pdb
import wandb
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from autoaugment import CIFAR10Policy

from datasets.cifar_dataset import cifar_dataset
from models import Encoder_Classifier
from losses.semi_loss import SemiLoss, linear_rampup
from losses.robust_losses import SCELoss
from losses.contrastive_loss import SupConLoss
from utils.utils import *
from utils.find_outliers import  find_outliers
from utils.seperate_lul_sets import seperate_lul_sets


def get_arguments():
    parser = argparse.ArgumentParser(
        description="SemiSupervised training of a pretrained model on CIFAR"
    )

    # Data
    parser.add_argument("--data-dir", default= "/datasets/cifar100/cifar-100-python/", type=Path, help="path to dataset")
    parser.add_argument("--num-class", default=100, type=int, help="number of classes")
    parser.add_argument('--corruption', type=str, default='imagenet32',
                        choices=['imagenet32', 'place365', 'cifar10', 'path'], help='dataset to corrupt cifar dataset (add ood samples)')
    parser.add_argument(
        "--noise-file",
        default="/datasets/cifar100_noisy_ood_02",
        type=Path,
        help="path to noise file",
    )
    parser.add_argument("--noise-ratio", type=float, default=0.2,
                        help='in-dist noise ratio')
    parser.add_argument("--ood-ratio", type=float, default=0.2,
                        help='ood noise ratio')
    parser.add_argument("--p-threshold", type=float, default=0.3,
                        help='GMM threshold')
    # Checkpoint
    parser.add_argument("--pretrained", type=Path, default='/save/last.pth', help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )
    parser.add_argument(
        "--gpu", default=0, type=int, metavar="N", help="GPU#"
    )
    # Model
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument(
        "--drops",
        default=0.005,
        type=float,
        metavar="percenatage of zero-out features",
    )
    parser.add_argument(
        "--weights",
        default="finetune",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )
    # Optim
    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        metavar="N",
        help="number of total epochs",
    )
    parser.add_argument(
        "--warmup-epoch",
        default=5,
        type=int,
        metavar="N",
        help="number of warmup training epochs",
    )
    parser.add_argument(
        "--interval",
        default=30,
        type=int,
        metavar="N",
        help="number of interval to apply GMM Clustering",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.03,
        type=float,
        metavar="LR",
        help="backbone learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classification head learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=5e-4, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="W", help="momentum"
    )
    parser.add_argument('--lambda_u', default=100, type=float, help='weight for unsupervised loss') #30 50 100
    parser.add_argument('--lambda_c', default=1, type=float, help='weight for contrastive loss') #0.025 0.5 0.5 
    parser.add_argument('--lambda_ce', default=1, type=float, help='weight for contrastive loss applied on embeddings')
    parser.add_argument('--lambda_ue', default=100, type=float, help='weight for contrastive loss applied on embeddings') #10 30 50 
    parser.add_argument('--lambda_x', default=1, type=float, help='weight for supervised loss')
    parser.add_argument('--lambda_xe', default=1, type=float, help='weight for supervised loss applied on embeddings')
    parser.add_argument('--temperature', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument(
        "--emb-loss",
        default="no",
        type=str,
        choices=("mse", "kl" , 'no'),
        help="types of loss applied on embeddings, Mean Squared Error, KL-Divergent, or nothing",
    )
    parser.add_argument(
        "--LN-loss",
        default="RL",
        type=str,
        choices=("CE", "RL"),
        help="types of loss to be robust to label noise, Cross-Entropy or Robust-Loss",
    )
    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    parser.add_argument(
        "--thr_un",
        default=0.9,
        type=float,
        metavar="df",
        help="threshold to accept unlabeled samples to use in training process",
    )
    parser.add_argument(
        "--remove-ood",
        default=0.01,
        type=float,
        metavar="df",
        help="Ration of removing OOD",
    )
    parser.add_argument(
        "--noise-type",
        default='sym',
        type=str,
        choices=("no", "sym", 'asym'),
        help="type of in-distribution label noise added",
    )
    parser.add_argument(
        "--clustering",
        default=False,
        type=bool,
        metavar="cluster",
        help="Apply GMM to find label noise data every interval?",
    )
    return parser

parser = get_arguments()
args = parser.parse_args()


def main():
    
    wandb.init(project="Manifold-test")

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.makedirs(args.exp_dir / 'images')
    except:
        print('File exist')

    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    
    chekpoint = torch.load(args.pretrained)
    sd = {}
    for ke in chekpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = chekpoint['model'][ke]
    model = Encoder_Classifier(args.arch, num_classes=args.num_class)
    model.load_state_dict(sd, strict=False)
    model.cuda(args.gpu)

    criterion_warmup = SCELoss(out_type='mean', num_classes=args.num_class).cuda(args.gpu)
    criterion = SemiLoss()
    contrastive_criterion = SupConLoss().cuda(args.gpu)
    # criterion_loss = NCEandRCE_mean(num_classes=10).cuda(args.gpu) 
    # if args.emb_loss =='mse':
    #     criterion_emb = nn.MSELoss().cuda(args.gpu)
    # else:
    #     criterion_emb = nn.KLDivLoss().cuda(args.gpu)
            
    encoder_p = []
    classifier_p = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_p.append(param)
        else:
            classifier_p.append(param)

    param_groups = [dict(params=classifier_p, lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params= model.encoder.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_acc_test = ckpt["best_acc"]
    else:
        start_epoch = 0
        best_acc_test = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    transform_weak_C100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )
    transform_strong_C100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )
    val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]
        )
    train_dataset = cifar_dataset(dataset='cifar100', noise_mode=args.noise_type, r=args.noise_ratio,  ood_noise=args.ood_ratio, root_dir=args.data_dir, transform=transform_weak_C100, mode="all_sup", noise_file=args.noise_file, corruption=args.corruption)                
    val_dataset = cifar_dataset(dataset='cifar100', noise_mode=args.noise_type, r=args.noise_ratio,  ood_noise=args.ood_ratio,  root_dir=args.data_dir, transform=val_transforms, mode="all_sup", noise_file=args.noise_file, corruption=args.corruption )                
    test_dataset = cifar_dataset(dataset='cifar100', noise_mode='no', r=args.noise_ratio,  ood_noise=args.ood_ratio, root_dir=args.data_dir, transform=val_transforms, mode="test", corruption=args.corruption)                

    kwargs = dict(
        batch_size=args.batch_size ,
        num_workers=args.workers,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    

    # Find ood for removing from training process (updating the weights of model) ...
    removal_ratio = int(np.ceil(args.remove_ood*10*len(train_dataset)))
    print(removal_ratio)
    ood_index = find_outliers(val_loader, model, args.gpu, removal_ratio)
    pdb.set_trace()
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        if epoch < args.warmup_epoch:
            #warmup training ...
            for step, (images, target, idx) in enumerate(
                train_loader, start=epoch * len(train_loader)
            ):
                mask_ood = torch.ones(idx.shape[0]).float().cuda()
                mask_ood = 1 - ood_index[idx]

                _,_, output = model(images.cuda(args.gpu, non_blocking=True))

                loss = (mask_ood*criterion_warmup(output, target.cuda(args.gpu, non_blocking=True), onehot=True)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % args.print_freq == 0:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

                    wandb.log({"TrainLoss": loss.item(), 'custom_step': epoch})
                    wandb.log({"LR_head": lr_head, 'custom_step': epoch})
                    wandb.log({"LR_backbone": lr_backbone, 'custom_step': epoch})
        else:
            # semi-supervised training using MixEMatch Algorithm
            if epoch == args.warmup_epoch or ( epoch % args.interval == 0 and args.clustering):
                
                removal_ratio = removal_ratio + int(np.ceil(args.remove_ood*len(train_dataset)))
                ood_index = find_outliers(val_loader, model, args.gpu, removal_ratio)
                mask_ood = 1 - ood_index.cpu().numpy()

                pred, prob = seperate_lul_sets(args, val_loader, model, epoch)

                labeled_dataset = cifar_dataset(dataset='cifar100', noise_mode=args.noise_type, r=args.noise_ratio, ood_noise=args.ood_ratio, root_dir=args.data_dir, transform=transform_weak_C100, mode="labeled", noise_file=args.noise_file, corruption=args.corruption, ood=mask_ood, pred=pred, probability=prob, transform_st=transform_strong_C100)
                unlabeled_dataset = cifar_dataset(dataset='cifar100', noise_mode=args.noise_type, r=args.noise_ratio,ood_noise=args.ood_ratio, root_dir=args.data_dir, transform=transform_weak_C100, mode="unlabeled", noise_file=args.noise_file, corruption=args.corruption, ood=mask_ood, pred=pred, transform_st=transform_strong_C100)                    
                
                labeled_trainloader = torch.utils.data.DataLoader(
                    labeled_dataset, **kwargs
                )
                unlabeled_trainloader = torch.utils.data.DataLoader(
                    unlabeled_dataset, **kwargs
                )
                
            unlabeled_train_iter = iter(unlabeled_trainloader)   
            num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
            for step, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader, start=epoch * len(labeled_trainloader)):      
                try:
                    inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
                except:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)                 
                batch_size = inputs_x.size(0)

                # Transform label to one-hot
                labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
                w_x = w_x.view(-1,1).type(torch.FloatTensor) 

                w_x = w_x.cuda(args.gpu, non_blocking=True)
                labels_x = labels_x.cuda(args.gpu, non_blocking=True)

                with torch.no_grad():
                    #Extracting embedding space
                    # label guessing of unlabeled samples
                    _,_, outputs_u11 = model(inputs_u.cuda(args.gpu, non_blocking=True))
                    _,_, outputs_u12 = model(inputs_u2.cuda(args.gpu, non_blocking=True))

                    
                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1))/2 
                    ptu = pu**(1/args.temperature) # temparature sharpening
                    
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                    targets_u = targets_u.detach()       
                    max_probs, _ = torch.max(targets_u, dim=-1)
                    mask = max_probs.ge(args.thr_un).float()
                    
                    ## Label refinement
                    _,_, outputs_x  = model(inputs_x.cuda(args.gpu, non_blocking=True))
                    _,_, outputs_x2 = model(inputs_x2.cuda(args.gpu, non_blocking=True))            
                
                    px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = w_x*labels_x + (1-w_x)*px              
                    ptx = px**(1/args.temperature) # temparature sharpening 
                            
                    targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
                    targets_x = targets_x.detach()   
                    mask_x = torch.ones(batch_size).float().cuda()
                
                
                emb_x3,_, _ = model(inputs_x3.cuda(args.gpu, non_blocking=True))
                emb_x4,_,_ = model(inputs_x4.cuda(args.gpu, non_blocking=True))

                emb_u3, f1, _ = model(inputs_u3.cuda(args.gpu, non_blocking=True))
                emb_u4,f2, _ = model(inputs_u4.cuda(args.gpu, non_blocking=True))

                f1 = F.normalize(f1, dim=1)
                f2 = F.normalize(f2, dim=1)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_simCLR = contrastive_criterion(features)
                
                # mixEmatch Algorithm
                all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
                all_emb =  torch.cat([emb_x3, emb_x4, emb_u3, emb_u4], dim=0)
                all_mask = torch.cat([mask_x, mask_x,mask, mask], dim=0)
                
                ## Apply Mixup on both inputs and embeddings
                idx = torch.randperm(all_inputs.size(0))
                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                emb_a, emb_b = all_emb, all_emb[idx]
                msk_a, msk_b = all_mask, all_mask[idx]
                ### generate multiplier, lambda, to mixed up inputs, labels, and embeddings
                la = np.random.beta(args.alpha, args.alpha)        
                la = max(la, 1-la)
                mixed_input = la * input_a + (1 - la) * input_b        
                mixed_target = la * target_a + (1 - la) * target_b
                mixed_emb = la * emb_a + (1 - la) * emb_b
                mixed_mask =  la * msk_a + (1 - la) * msk_b


                _, f3, logits = model(mixed_input.cuda(args.gpu, non_blocking=True))
                _, f4, logits_emb = model(mixed_emb.cuda(args.gpu, non_blocking=True), 'emb')   

                ## Unsupervised Contrastive Loss for embedding space
                # f3 = F.normalize(f3, dim=1)
                # f4 = F.normalize(f4, dim=1)
                # features_emb = torch.cat([f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)
                # loss_simCLR_emb = contrastive_criterion(features_emb) 

                logits_x = logits[:batch_size*2]
                logits_u = logits[batch_size*2:]   
                logits_emx = logits_emb[:batch_size*2]
                logits_emu = logits_emb[batch_size*2:]      
                mask_x = mixed_mask[:batch_size*2]
                mask_u = mixed_mask[batch_size*2:]   

                ## Combined Loss
                Lx, Lu = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], mask_x, mask_u, args.num_class, args.gpu, args.LN_loss)
                Lxe, Lue = criterion(logits_emx, mixed_target[:batch_size*2], logits_emu, mixed_target[batch_size*2:], mask_x, mask_u, args.num_class, args.gpu, args.LN_loss)

                lamb = args.lambda_u * linear_rampup( epoch+step/num_iter, args.warmup_epoch)
                lambe = args.lambda_ue * linear_rampup( epoch+step/num_iter, args.warmup_epoch)

                ## Regularization
                prior = torch.ones(args.num_class)/args.num_class
                prior = prior.cuda()        
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))

                ## Total Loss               
                loss = args.lambda_x * Lx +  lamb *Lu  +args.lambda_xe * Lxe + lambe * Lue  + args.lambda_c*loss_simCLR +  penalty 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.print_freq == 0:                     
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    wandb.log({"LX": Lx, 'custom_step': epoch})
                    wandb.log({"Lu": Lu, 'custom_step': epoch})
                    wandb.log({"LXe": Lxe, 'custom_step': epoch})
                    wandb.log({"Lue": Lue, 'custom_step': epoch})
                    wandb.log({"Co": loss_simCLR, 'custom_step': epoch})
                    wandb.log({"TrainLoss": loss.item(), 'custom_step': epoch})
                    wandb.log({"LR_head": lr_head, 'custom_step': epoch})
                    wandb.log({"LR_backbone": lr_backbone, 'custom_step': epoch})

        # evaluate
        model.eval()
        save_best = False
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")
        with torch.no_grad():
            for images, target, _ in val_loader:
                _,_, output = model(images.cuda(args.gpu, non_blocking=True))
                acc1, acc5 = accuracy(
                    output, target.cuda(args.gpu, non_blocking=True), topk=(1, 5)
                )
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
        if best_acc.top1 < top1.avg:
            save_best = True
        best_acc.top1 = max(best_acc.top1, top1.avg)
        best_acc.top5 = max(best_acc.top5, top5.avg)
        wandb.log({"Top1": top1.avg, 'custom_step': epoch})
        wandb.log({"Top5": top5.avg, 'custom_step': epoch})
        wandb.log({"BestTop1": best_acc.top1, 'custom_step': epoch})
        wandb.log({"BestTop5": best_acc.top5, 'custom_step': epoch})

        stats = dict(
            epoch=epoch,
            acc1=top1.avg,
            acc5=top5.avg,
            best_acc1=best_acc.top1,
            best_acc5=best_acc.top5,
        )
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

        scheduler.step()
        
        
        test_top1 = AverageMeter("Acc@1")
        test_top5 = AverageMeter("Acc@5")
        with torch.no_grad():
            for images, target in test_loader:
                _,_, output = model(images.cuda(args.gpu, non_blocking=True))
                test_acc1, test_acc5 = accuracy(
                    output, target.cuda(args.gpu, non_blocking=True), topk=(1, 5)
                )
                test_top1.update(test_acc1[0].item(), images.size(0))
                test_top5.update(test_acc5[0].item(), images.size(0))
        best_acc_test = max(best_acc_test, test_top1.avg)

        state = dict(
            epoch=epoch + 1,
            best_acc=best_acc,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best_acc_test = best_acc_test
        )
        if save_best: 
            if epoch < args.warmup_epoch:
                torch.save(state, args.exp_dir / "best_checkpoint_w.pth")
                save_best = False
            elif epoch == args.warmup_epoch:
                torch.save(state, args.exp_dir / "best_checkpoint_warmup.pth")
                save_best = False
            else:
                torch.save(state, args.exp_dir / "best_checkpoint.pth")
                save_best = False
        elif epoch==args.warmup_epoch-1:
            torch.save(state, args.exp_dir / "checkpoint_before_semi.pth")
        else:
            torch.save(state, args.exp_dir / "checkpoint.pth")

        if best_acc_test < test_top1.avg:
            torch.save(state, args.exp_dir / "besttest_checkpoint.pth")
        wandb.log({"Test Top1": test_top1.avg, 'custom_step': epoch})
        wandb.log({"Test Top5": test_top5.avg, 'custom_step': epoch})

    torch.save(model.state_dict(), args.exp_dir / "last_resnet18.pth")
                    
    # wandb.finish()





if __name__ == "__main__":
    main()
