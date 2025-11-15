from typing import Iterable
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import util.misc as utils
from util.misc import normalize,renormalize,constrained_bezier
import functools
from tqdm import tqdm
import torch.nn.functional as F
from monai.metrics import compute_dice
# compute_dice
from util.misc import constrained_bezier
import torch.nn as nn
from torch.autograd import Variable
# from clip import clip
import wandb
from sklearn.manifold import TSNE
from sklearn import preprocessing

import cv2
from dataloaders.saliency_balancing_fusion import get_SBF_map
from dataloaders.image_transforms import fourier_augmentation_for_tensor

print = functools.partial(print, flush=True)

# def compute_discrepancy(
#     features_T, features_S, 
# ):
#     normalized_features_T, normalized_features_S,  = features_T/torch.norm(features_T, p = 2), features_S/torch.norm(features_S, p = 2), 
#     discrepancy = torch.square(torch.dist(
#         normalized_features_T, normalized_features_S, 
#         p = 2, 
#     ))
#     return discrepancy -(nn.CosineSimilarity(dim=-1, eps=1e-6)(normalized_features_T, normalized_features_S)).mean()
def compute_discrepancy(
    features_T, features_S, 
):
    return -(nn.CosineSimilarity(dim=-1, eps=1e-6)(features_T, features_S)).mean()
def p(x):
    return 1/(1+torch.exp(x))+0.000001
def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, learning_rate:float, warmup_iteration: int = 1500):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 10
    cur_iteration=0
    while True:
        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, 'WarmUp with max iteration: {}'.format(warmup_iteration))):
            for k,v in samples.items():
                if isinstance(samples[k],torch.Tensor):
                    samples[k]=v.to(device)
            cur_iteration+=1
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = cur_iteration/warmup_iteration*learning_rate * param_group["lr_scale"]

            img=samples['images']
            lbl=samples['labels']
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if cur_iteration>=warmup_iteration:
                print(f'WarnUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
                return cur_iteration
        metric_logger.synchronize_between_processes()
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1, grad_scaler=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        img = samples['images']
        lbl = samples['labels']

        if grad_scaler is None:
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss_dict = criterion.get_loss(pred,lbl)
                losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            grad_scaler.scale(losses).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cur_iteration+=1
        if cur_iteration>=max_iteration and max_iteration>0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration
def train_one_epoch_SBF(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,opt_aug:torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1,config=None,LGAug_config=None,\
                visdir=None,log=None,teacher_model=None,IIA_model=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    visual_freq = 1000
    t = torch.tensor(np.linspace(0, 1, 100)).cuda()
    FR_aug= fourier_augmentation_for_tensor(wandb.config.fda_beta)
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        GLA_img = samples['images']
        LLA_img = samples['aug_images']
        lbl = samples['labels']
        raw_image=samples['raw_images']
        index =samples['index']
        if cur_iteration % visual_freq == 0:
            visual_dict={}
            visual_dict['GLA']=GLA_img.detach().cpu().numpy()[0,0]
            visual_dict['LLA']=LLA_img.detach().cpu().numpy()[0,0]
            visual_dict['GT']=lbl.detach().cpu().numpy()[0]
        else:
            visual_dict=None
            
        # GLA_img = torch.tensor(FR_aug(GLA_img.detach().cpu().numpy()[:,0,:,:])[:,None,:,:]).cuda().float()
        input_var = Variable(GLA_img, requires_grad=True)

        optimizer.zero_grad()
        output = model(input_var)
        logits = output["logits"]
        loss_dict = criterion.get_loss(logits, lbl)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        losses.backward(retain_graph=True)

        with torch.no_grad():
            gradient = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()
            saliency=get_SBF_map(gradient,config.grid_size)
        if visual_dict is not None:
            visual_dict['GLA_pred']=torch.argmax(logits,1).cpu().numpy()[0]
        if visual_dict is not None:
            visual_dict['GLA_saliency']= saliency.detach().cpu().numpy()[0,0]
        mixed_img = GLA_img.detach() * saliency + LLA_img * (1 - saliency)
        if visual_dict is not None:
            visual_dict['SBF']= mixed_img.detach().cpu().numpy()[0,0]
            
        # mixed_img = torch.tensor(FR_aug(mixed_img.detach().cpu().numpy()[:,0,:,:])[:,None,:,:]).cuda().float()
        aug_var = Variable(mixed_img, requires_grad=True)
        aug_logits = model(aug_var)["logits"]
        aug_loss_dict = criterion.get_loss(aug_logits, lbl)
        aug_losses = sum(aug_loss_dict[k] * criterion.weight_dict[k] for k in aug_loss_dict.keys() if k in criterion.weight_dict)
        aug_losses.backward()

        if visual_dict is not None:
            visual_dict['SBF_pred'] = torch.argmax(aug_logits, 1).cpu().numpy()[0]
        optimizer.step()

        if(LGAug_config.image_aug):
            opt_aug.zero_grad()
            optimizer.zero_grad()
            output= IIA_model(raw_image.clone().detach(),model,lbl,index,cur_iteration,visual_freq,visual_dict)
            mixed_img = raw_image.detach() * saliency + output["IIA_img_aug"]* (1 - saliency)    
            if visual_dict is not None:
                visual_dict['LLA_img_new']=output["IIA_img_aug"].detach().cpu().numpy()[0,0]
                visual_dict['mixed_img']=mixed_img.detach().cpu().numpy()[0,0]
            with torch.no_grad():  
                features_T = teacher_model(raw_image.float())["fine_feature"]
            features_S = model(mixed_img.float())["fine_feature"]
            discrepancy = compute_discrepancy(
                    features_T, features_S, 
                )
            loss_FS = criterion.get_loss(model.base_model.segmentation_head(features_S), lbl)
            loss_FS = sum(loss_FS[k] * criterion.weight_dict[k] for k in loss_FS.keys() if k in criterion.weight_dict)
            loss_FS = loss_FS + (LGAug_config.loss_aug_param + LGAug_config.consistent_param)*discrepancy
            loss_GS = - LGAug_config.loss_aug_param*discrepancy
            for parameter in IIA_model.parameters():
                parameter.requires_grad = False
            loss_FS.backward(retain_graph = True)
            for parameter in IIA_model.parameters():
                parameter.requires_grad = True
            loss_GS.backward(retain_graph = False)
            # Domain Generalized Representation Learning
            optimizer.step()
            # Learning to Generate Novel Domains
            opt_aug.step()
            # data_loader.dataset.transforms
            mixed_img = torch.tensor(FR_aug(mixed_img.detach().cpu().numpy()[:,0,:,:])[:,None,:,:]).cuda().float()
            output = model(mixed_img)
            loss_FS = criterion.get_loss(output["logits"], lbl)
            loss_FS = sum(loss_FS[k] * criterion.weight_dict[k] for k in loss_FS.keys() if k in criterion.weight_dict)
            loss_FS.backward(retain_graph = False)
            optimizer.step()
            
            state_dict_FT, state_dict_FS,  = teacher_model.state_dict(), model.state_dict(), 
            for parameter in state_dict_FS:
                state_dict_FT[parameter] = 0.999*state_dict_FT[parameter] + (1 - 0.999)*state_dict_FS[parameter]
            teacher_model.load_state_dict(state_dict_FT)
            
            optimizer.zero_grad()
            opt_aug.zero_grad()

        all_loss_dict={}
        for k in loss_dict.keys():
            if k not in criterion.weight_dict:continue
            all_loss_dict[k]=loss_dict[k]
            all_loss_dict[k+'_aug']=aug_loss_dict[k]
            
        # if (LGAug_config.image_aug):    
        #     all_loss_dict['loss_var']=loss_var
        # wandb.log({"losses":losses,"aug_losses":aug_losses,"loss_var":loss_var,})
        metric_logger.update(**all_loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if cur_iteration>=max_iteration and max_iteration>0:
            break

        if visdir is not None and cur_iteration%visual_freq == 0:
            # log.logger.info((p(model.param))[index,i,t,layer,4])
            fs=int(len(visual_dict)**0.5)+1
            fig = plt.figure(figsize=(12, 12), dpi=400)
            for idx, k in enumerate(visual_dict.keys()):
                plt.subplot(fs,fs,idx+1)
                plt.title(k)
                plt.axis('off')
                if "plot" in k:
                    plt.axis('on')
                    plt.plot(t.cpu().detach().numpy(), visual_dict[k], '-b')
                    
                elif k not in ['GT','GLA_pred','SBF_pred','logits_bg','aug_logits_bg_aug',"mixed_img_pred","mixed_img_pred_new",]:
                    plt.axis('off')
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.axis('off')
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
            plt.tight_layout()
            plt.savefig(f'{visdir}/{cur_iteration}.png')
            Img= wandb.Image(plt, caption=f'{cur_iteration}.png')
            wandb.log({f'visualization':Img,})
            plt.close()
            
        cur_iteration+=1

    metric_logger.synchronize_between_processes()
    log.logger.info(metric_logger)
    print("Averaged stats:", metric_logger)
    return cur_iteration

def imshow_expand(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,target_data_loader: Iterable, device: torch.device, epoch: int, cur_iteration:int,SBF_config,config,IIA_model=None):
    model.train()
    source_feature= []
    source_expand_feature = []
    target_feature= []
    for i, samples in enumerate(data_loader):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        lbl = samples['labels']
        raw_image=samples['curr_dict']
        index =samples['index']
        input_var = Variable(raw_image, requires_grad=True)
        output = model(input_var)
        logits = output["logits"]
        loss_dict = criterion.get_loss(logits, lbl)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        losses.backward(retain_graph=True)
        source_feature.append(output["coarse_feature"].flatten(1).cpu().detach())
        # saliency
        with torch.no_grad():
            gradient = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()
            saliency = get_SBF_map(gradient, SBF_config.grid_size)

            IIA_img_aug, loss_var= IIA_model(raw_image.clone().detach(),model,lbl,index,1,1,None)
            mixed_img = raw_image.detach() * saliency + IIA_img_aug* (1 - saliency) 
            mixed_output = model(mixed_img)
            source_expand_feature.append(mixed_output["coarse_feature"].flatten(1).cpu().detach())
        # for i in range(len(mixed_img)):
        #     np.save(f"mydata/{config.data.params.train.params.modality[0]}expand/{index[i]}",mixed_img.detach().cpu().numpy()[i,0])
        #     plt.axis('off')
        #     plt.xticks([]) # 去刻度
        #     plt.imshow( mixed_img.detach().cpu().numpy()[i,0], cmap="gray")
        #     plt.savefig(f"mydata/{config.data.params.train.params.modality[0]}expand/{index[i]}.jpg",bbox_inches="tight", pad_inches = -0.1)
        #     plt.close()
    with torch.no_grad():        
        for i, samples in enumerate(target_data_loader):
            for k, v in samples.items():
                if isinstance(samples[k], torch.Tensor):
                    samples[k] = v.to(device)

            lbl = samples['labels']
            raw_image=samples['curr_dict']
            index =samples['index']
            input_var = Variable(raw_image, requires_grad=True)
            output = model(input_var)
            target_feature.append(output["coarse_feature"].flatten(1).cpu().detach())
            
    source_feature = torch.stack(source_feature).flatten(0,1)
    source_expand_feature = torch.stack(source_expand_feature).flatten(0,1)
    target_feature = torch.stack(target_feature).squeeze(1)
    
    label_MRI = torch.zeros(len(source_feature)) 
    label_MRI_expand = torch.ones(len(source_expand_feature)) 
    label_CT = torch.ones(len(target_feature))+1
    label = np.array(torch.cat([label_MRI,label_MRI_expand,label_CT]))
    tsne = TSNE(n_components=2, verbose=1 ,random_state=23)
    with torch.no_grad():
        # 获取文本和图像特征，这里其实没有用到，而是直接获取到了logits，也就是缩放后的余弦相似度结果
        CHAO_image_features = source_feature
        CHAO_expand_image_features = source_expand_feature

        SABSCT_image_features = target_feature

        image_features = torch.cat([CHAO_image_features,CHAO_expand_image_features, SABSCT_image_features],0)
        # image_features = image_features-image_features.mean(0)
        features =  image_features
        result = tsne.fit_transform(features.cpu())
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)
        plt.figure(figsize=(10, 10))
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.1, 1.1))
        plt.axis('off')
        plt.xticks([]) # 去刻度
        color=["r","g","b","c"]
        for i in range(len(result)):
            plt.text(result[i,0], result[i,1], str(label[i]), 
                    color=color[int(label[i])], fontdict={'weight': 'bold', 'size': 9})
        plt.savefig(f"mydata/{config.data.params.train.params.modality[0]}expand/tsne.jpg",bbox_inches="tight", pad_inches = -0.1)
        plt.close()
    return cur_iteration

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    def convert_to_one_hot(tensor,num_c):
        return F.one_hot(tensor,num_c).permute((0,3,1,2))
    dices=[]
    for samples in data_loader:
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)
        img = samples['images']
        lbl = samples['labels']
        logits = model(img)["logits"]
        num_classes=logits.size(1)
        pred=torch.argmax(logits,dim=1)
        one_hot_pred=convert_to_one_hot(pred,num_classes)
        one_hot_gt=convert_to_one_hot(lbl,num_classes)
        dice=compute_dice(one_hot_pred,one_hot_gt,include_background=False)
        dices.append(dice.cpu().numpy())
    dices=np.concatenate(dices,0)
    dices=np.nanmean(dices,0)
    return dices

def prediction_wrapper(model, test_loader, epoch, label_name, mode = 'base', save_prediction = False):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    model.eval()
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        # recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['images'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['labels'].shape[0] == 1 # enforce a batchsize of 1

            img = batch['images'].cuda()
            gth = batch['labels'].cuda()

            pred = model(img)["logits"]
            pred = torch.argmax(pred,1)
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['images'][0, 0,...].numpy()
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                out_prediction_list[scan_id_full]['img'] = curr_img

                # if opt.phase == 'test':
                #     recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name),label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()
 
    return out_prediction_list, dsc_table, error_dict, domain_names


def prediction_wrapper2(student,teacher, test_loader, epoch, label_name, mode = 'base', save_prediction = False):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    student.eval()
    teacher.eval()

    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        # recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['images'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['labels'].shape[0] == 1 # enforce a batchsize of 1

            img = batch['images'].cuda()
            gth = batch['labels'].cuda()

            pred = (student(img)["logits"]+teacher(img)["logits"])/2.0
            pred = torch.argmax(pred,1)
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['images'][0, 0,...].numpy()
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                out_prediction_list[scan_id_full]['img'] = curr_img

                # if opt.phase == 'test':
                #     recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name),label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()
 
    return out_prediction_list, dsc_table, error_dict, domain_names


def eval_list_wrapper(vol_list, nclass, label_name):
    """
    Evaluatation and arrange predictions
    """
    def convert_to_one_hot2(tensor,num_c):
        return F.one_hot(tensor.long(),num_c).permute((3,0,1,2)).unsqueeze(0)

    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures
    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices=compute_dice(y_pred=convert_to_one_hot2(pred_,nclass),y=convert_to_one_hot2(gth_,nclass),include_background=True).cpu().numpy()[0].tolist()

        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
    print("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)
    print('per domain resutls:', overall_by_domain)
    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    return error_dict, dsc_table, domain_names

