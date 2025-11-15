import argparse, os, sys, datetime, importlib
os.environ['KMP_DUPLICATE_LIB_OK']='true'
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys
sys.path.append("./")
from engine import train_warm_up,evaluate,train_one_epoch_SBF,train_one_epoch,prediction_wrapper,prediction_wrapper2,normalize,constrained_bezier,renormalize,p,imshow_expand
from losses import SetCriterion
import numpy as np
import random
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from logging import handlers
import logging
import torch.nn as nn
import copy
import wandb 
import cv2


class IIA(nn.Module):
    def __init__(self,data,config):
        super(IIA, self).__init__()
        self.conv_coarse=nn.Conv2d(352, 128, 3, stride=2)
        self.data_num = len(data.datasets["train"])
        self.aug_num = config.aug_num   #不同变换的数量
        self.cnum = len(data.datasets["train"].all_label_names)    #类别的数量
        self.lnum = config.layer_num    #层的数量
        self.t = torch.tensor(np.linspace(0, 1, 100)).cuda()
        self.conv_fine=nn.Conv2d(16, 512, 3, stride=1, padding=1)
        self.mix_param=nn.Parameter(torch.rand(self.data_num,self.cnum,self.aug_num,self.lnum,1,1))
        self.param=nn.Parameter(torch.rand(self.data_num,self.cnum,self.aug_num,self.lnum,5+2))
        self.random_conv = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Sequential(
              nn.Conv2d(1, 4, 3, stride=1,padding=1),
              nn.Conv2d(4, 1, 3, stride=1,padding=1),
                ) for _ in range(self.lnum)]) for _ in range(self.aug_num)]) for _ in range(self.cnum)]) for _ in range(self.data_num)])
        # self.random_conv_b = nn.Parameter(torch.rand(data_num,self.cnum,self.aug_num,self.lnum,2))
    def forward(self,GLA_img_aug,model,lbl,index,cur_iteration,visual_freq,visual_dict):
        GLA_img_raw = GLA_img_aug.detach().cpu().numpy()
        pselect = np.random.randint(0, self.param.shape[-3])
        GLA_img_aug,_,_,img_mean,img_std = normalize(GLA_img_aug) 
        GLA_img_ct_res = 0
        GLA_img_cv_res = 0
        mix_fusion_res = 0
        label0= torch.unique(lbl[0])
        for i in range(self.param.shape[-4]):
            if (i==0):
                GLA_img_aug, plist = self.Ftransfer(GLA_img_aug,lbl,i,1,index,t=pselect)
            else:
                GLA_img_aug, plist = self.Ftransfer(GLA_img_aug,lbl,i,0.5,index,t=pselect)
            if(i in label0):
                GLA_img_ct=plist[8][0].detach().cpu().numpy()
                GLA_img_cv=plist[9][0].detach().cpu().numpy()
                mix=plist[10][0].detach().cpu().numpy()     
                GLA_img_ct_res += GLA_img_ct[0]
                GLA_img_cv[lbl[0].detach().cpu().numpy()[None,:,:]!=i]=0
                GLA_img_cv_res += GLA_img_cv
                mix[lbl.detach().cpu().numpy()[0][None,:,:]!=i]=0
                mix_fusion_res += mix[0] 
                
        if cur_iteration % visual_freq == 0:
            if(len(label0)>0):
                B=constrained_bezier(plist[0], plist[1][0], plist[2][0], plist[3])
                Bv=constrained_bezier(plist[4], plist[5][0], plist[6][0], plist[7])
                if visual_dict is not None:
                    # visual_dict[f'plot{i}_{mix:.2}']=B(t).cpu().detach().numpy()    
                    # visual_dict[f'plotv{i}_{1-mix:.2}']=Bv(t).cpu().detach().numpy()
                    visual_dict[f'plot{i}']=B(self.t).cpu().detach().numpy()    
                    visual_dict[f'plotv{i}']=Bv(self.t).cpu().detach().numpy()
                    visual_dict[f'GLA_img_ct']=GLA_img_ct_res
                    visual_dict[f'GLA_img_cv']=GLA_img_cv_res[0]
                    visual_dict[f'mix_fusion']=mix_fusion_res      
                             
        GLA_img_aug = renormalize(GLA_img_aug,None,None,img_mean,img_std) 
        return {"IIA_img_aug":GLA_img_aug,}
    
    def Ftransfer(self,GLA_img_aug,lbl,i,ratio,index,layer=0,t=0):
        GLA_img_raw = GLA_img_aug.clone()
        lbl_new= (lbl[:,None,:,:]==i)
        GLA_img_c = GLA_img_aug*lbl_new
        flag = (torch.sum(torch.sum(lbl_new,-1),-1)>0).flatten()
        GLA_img_c,lbl_new = GLA_img_c[flag],lbl_new[flag]
        p0,p3,p0v,p3v= 0,1,1,0
        for layer in range(self.param.shape[-2]):
            p1  = (p(self.param))[index,i,t,layer,0][flag]
            p2  = (p(self.param))[index,i,t,layer,1][flag]
            p1v = (p(self.param))[index,i,t,layer,2][flag]
            p2v = (p(self.param))[index,i,t,layer,3][flag]
            mix= [self.random_conv[inde][i][t][layer] for inde in index[flag]]
            mix = p(torch.stack([mix[i](GLA_img_raw[flag][i]) for i in range(len(mix))]))
            GLA_img_ct = (1-GLA_img_c)**3 * p0  + 3*(1-GLA_img_c)**2*GLA_img_c*(p1[:,None,None,None]) + 3*(1-GLA_img_c)*GLA_img_c**2*(p2[:,None,None,None] )+ GLA_img_c**3*p3
            GLA_img_cv = (1-GLA_img_c)**3 * p0v + 3*(1-GLA_img_c)**2*GLA_img_c*(p1v[:,None,None,None])+ 3*(1-GLA_img_c)*GLA_img_c**2*(p2v[:,None,None,None])+ GLA_img_c**3*p3v
            GLA_img_c =  GLA_img_ct*mix+ GLA_img_cv*(1 - mix) 
            GLA_img_c = torch.clamp(GLA_img_c, 0, 1)
        GLA_img_raw[lbl[:,None,:,:]==i]=GLA_img_c[lbl[flag][:,None,:,:]==i]
        return GLA_img_raw, [p0, p1, p2, p3, p0v, p1v, p2v, p3v, GLA_img_ct, GLA_img_cv, mix]

class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        
    def forward(self,x):
        encoder_outputs = self.base_model.encoder(x) #length 6 0=(32,1,192,192),  1=(32,32,96,96), 2=(32,24,48,48), 3=(32,48,24,24), 4=(32,120,12,12), 5=(32,352,6,6)
        enc_reprn = encoder_outputs[-1]
        flat_enc= enc_reprn.view(enc_reprn.size(0), -1, enc_reprn.size(1))   # Flatten the output to (batch_size, 36 * 352)
        dec_output = self.base_model.decoder(*encoder_outputs)
        output = self.base_model.segmentation_head(dec_output)
        # proj = self.linear(flat_enc)  # Project to (batch_size, 36 * 512)    
        return {"logits": output, "coarse_feature": enc_reprn,"fine_feature": dec_output,"flat_enc": flat_enc}
    
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
        
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'training seed is {seed}')
    return seed

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    return parser

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class DataModuleFromConfig(torch.nn.Module):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed=seed_everything(opt.seed)
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name=None
        raise ValueError('no config')

    nowname = now +f'_seed{seed}'+ name + opt.postfix
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    visdir= os.path.join(logdir, "visuals")
    for d in [logdir, cfgdir, ckptdir,visdir ]:
        os.makedirs(d, exist_ok=True)
    log = Logger(os.path.join(logdir, "experiment.log"),level='info')

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    OmegaConf.save(config,os.path.join(cfgdir, "{}-project.yaml".format(now)))

    model_config = config.pop("model", OmegaConf.create())
    optimizer_config = config.pop('optimizer', OmegaConf.create())
    SBF_config = config.pop('saliency_balancing_fusion',OmegaConf.create())
    LGAug_config = config.pop('LGAug',OmegaConf.create())
    
    wandb.init(project="Ki_"+config.data.params.train.params.modality[0]+"2"+config.data.params.test.params.modality[0], name=config.data.params.train.params.modality[0]+"2"+config.data.params.test.params.modality[0])
    for i in wandb.config:
        print("load", i, wandb.config[i])
        LGAug_config[i]=wandb.config[i]
    wandb.config.update(OmegaConf.to_container(LGAug_config),allow_val_change=True)
    wandb.config.update(OmegaConf.to_container(optimizer_config),allow_val_change=True)

    # model = instantiate_from_config(model_config)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    model = CustomModel(instantiate_from_config(model_config))
    teacher = CustomModel(instantiate_from_config(model_config))
    IIA_model = IIA(data, LGAug_config)

    if torch.cuda.is_available():
        model = model.cuda()
        teacher = teacher.cuda()
        IIA_model = IIA_model.cuda()

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        pl_sd = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(pl_sd['model'], strict=True)
        IIA_model.load_state_dict(pl_sd['IIA_model'], strict=True)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs, lr = config.data.params.batch_size, optimizer_config.learning_rate

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and "base_model" in n], "lr_scale": 1, 'lr': optimizer_config.learning_rate}]
    param_dicts_aug = [{"params": [p for n, p in IIA_model.named_parameters() if p.requires_grad and "random_conv" not in n], "lr_scale": 1,'lr': LGAug_config.learning_rateforaug},
                       {"params": [p for n, p in IIA_model.named_parameters() if p.requires_grad and "random_conv" in n], "lr_scale": 1,'lr': LGAug_config.learning_rateforaug2}]
    opt_params = {}
    opt_params_aug = {}
    for k in ['momentum', 'weight_decay']:
        if k in optimizer_config:
            opt_params[k] = optimizer_config[k]

    criterion = SetCriterion()

    print('optimization parameters: ', opt_params)
    opt = eval(optimizer_config['target'])(param_dicts, **opt_params)
    opt_aug = eval(optimizer_config['target'])(param_dicts_aug, **opt_params_aug)

    if optimizer_config.lr_scheduler =='lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 0 - 50) / float(optimizer_config.max_epoch-50 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)
        scheduler_aug = lr_scheduler.LambdaLR(opt_aug, lr_lambda=lambda_rule)

    else:
        scheduler=None
        scheduler_aug=None
        print('We follow the SSDG learning rate schedule by default, you can add your own schedule by yourself')
        raise NotImplementedError

    assert optimizer_config.max_epoch > 0 or optimizer_config.max_iter > 0
    if optimizer_config.max_iter > 0:
        max_epoch=999
        print('detect identified max iteration, set max_epoch to 999')
    else:
        max_epoch= optimizer_config.max_epoch

    #data.num_workers
    print(len(data.datasets["train"]))
    train_loader = DataLoader(data.datasets["train"], batch_size=data.batch_size,
                          num_workers=data.num_workers, shuffle=True, persistent_workers=True, drop_last=True, pin_memory = True)

    val_loader = DataLoader(data.datasets["validation"], batch_size=data.batch_size,  num_workers=1)

    if data.datasets.get('test') is not None:
        test_loader = DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
        best_test_dice = 0
        test_phase = True
    else:
        test_phase=False

    if getattr(optimizer_config, 'warmup_iter'):
        if optimizer_config.warmup_iter>0:
            train_warm_up(model, criterion, train_loader, opt, torch.device('cuda'), lr, optimizer_config.warmup_iter)
    cur_iter=0
    best_dice=0
    T_best_dice=0
    best_dice_val=0
    T_best_dice_val=0
    
    label_name=data.datasets["train"].all_label_names
    for cur_epoch in range(max_epoch):
        if SBF_config.usage:
            cur_iter = train_one_epoch_SBF(model, criterion,train_loader,opt,opt_aug, torch.device('cuda'),cur_epoch,cur_iter, optimizer_config.max_iter, SBF_config,LGAug_config, visdir,log,teacher,IIA_model)
        else:
            cur_iter = train_one_epoch(model, criterion, train_loader, opt, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter)
        if scheduler is not None:
            scheduler.step()
            scheduler_aug.step()
        # Save Best model on val
        if (cur_epoch)%50==0:
            log.logger.info("save at:"+ ckptdir)
#********************************************************************************************************
#student evaluate in source domain
#********************************************************************************************************
            cur_dice = evaluate(model, val_loader, torch.device('cuda'))
            if np.mean(cur_dice)>best_dice:
                best_dice=np.mean(cur_dice)
                # for f in os.listdir(ckptdir):
                #     if 'val' in f:
                #         os.remove(os.path.join(ckptdir,f))
                # torch.save({'model': model.state_dict()}, os.path.join(ckptdir,f'val_best_epoch_{cur_epoch}.pth'))
            str=f'Epoch [{cur_epoch}]   '
            for i,d in enumerate(cur_dice):
                str+=f'Class {i}: {d}, '
            str+=f'Validation DICE {np.mean(cur_dice)}/{best_dice}'
            print(str)
            wandb.log({"cur_dice":np.mean(cur_dice),"best_dice":best_dice})

#********************************************************************************************************
#student evaluate in target domain
#********************************************************************************************************
            out_prediction_list, dsc_table, error_dict, domain_names = prediction_wrapper(model, test_loader, 0, label_name, save_prediction=True)
            for organ in range(len(label_name)):
                mean_dc = np.mean(dsc_table[:, organ])
                std_dc  = np.std(dsc_table[:, organ])
                log.logger.info("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
            log.logger.info("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
            log.logger.info("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
            cur_dice_val = error_dict['overall_by_domain']
            if np.mean(cur_dice_val) > best_dice_val:
                best_dice_val = np.mean(cur_dice_val)
                # for f in os.listdir(ckptdir):
                #     if 'student_best_epoch' in f:
                #         os.remove(os.path.join(ckptdir,f))
                torch.save({'model': model.state_dict(),'IIA_model': IIA_model.state_dict()}, os.path.join(ckptdir,f'student_best_epoch.pth'))
            wandb.log({"cur_dice_val":np.mean(cur_dice_val),"best_dice_val":best_dice_val})
#********************************************************************************************************
#teacher evaluate in target domain
#********************************************************************************************************
            log.logger.info("teacher performance")
            T_cur_dice = evaluate(teacher, val_loader, torch.device('cuda'))
            if np.mean(T_cur_dice)>T_best_dice:
                T_best_dice=np.mean(T_cur_dice)
            T_str=f'Epoch [{cur_epoch}]   '
            for i,d in enumerate(T_cur_dice):
                T_str+=f'Class {i}: {d}, '
            T_str+=f'Validation teacher DICE {np.mean(T_cur_dice)}/{T_best_dice}'
            print(T_str)
            T_out_prediction_list, T_dsc_table, T_error_dict, T_domain_names = prediction_wrapper2(model,teacher, test_loader, 0, label_name, save_prediction=True)
            for organ in range(len(label_name)):
                T_mean_dc = np.mean(T_dsc_table[:, organ])
                T_std_dc  = np.std(T_dsc_table[:, organ])
                log.logger.info("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], T_mean_dc, T_std_dc))
            log.logger.info("Overall std dice by sample {:06.5f}".format(T_dsc_table[:, 1:].std()))
            log.logger.info("Overall mean dice by sample {:06.5f}".format( T_dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
            T_cur_dice_val = T_error_dict['overall_by_domain']
            if np.mean(T_cur_dice_val) > T_best_dice_val:
                T_best_dice_val = np.mean(T_cur_dice_val)
                # for f in os.listdir(ckptdir):
                #     if 'teacher_best_epoch' in f:
                #         os.remove(os.path.join(ckptdir,f))
                torch.save({'model': teacher.state_dict(),'IIA_model': IIA_model.state_dict()}, os.path.join(ckptdir,f'teacher_best_epoch.pth'))                
            wandb.log({"T_cur_dice_val":np.mean(T_cur_dice_val),"T_best_dice_val":T_best_dice_val})

                # for f in os.listdir(ckptdir):
                #     if 'test' in f:
                #         os.remove(os.path.join(ckptdir,f))
                # torch.save({'model': model.state_dict()}, os.path.join(ckptdir,f'test_best_epoch_{cur_epoch}.pth'))
                # torch.save({'model_base_model': model.base_model.state_dict()}, os.path.join(ckptdir,f'test_best_epoch_basemodel_{cur_epoch}.pth'))

#********************************************************************************************************
            # for i,d in enumerate(cur_dice_val):
            #     str+=f'Class {i}: {d}, '
            str+=f' Test DICE {np.mean(cur_dice_val)}/{best_dice_val} Teacher: {np.mean(T_cur_dice_val)}/{T_best_dice_val}'            
            log.logger.info(str)
            

    model.load_state_dict(torch.load(os.path.join(ckptdir,f'student_best_epoch.pth'), map_location="cpu")['model'], strict=False)
    out_prediction_list, dsc_table, error_dict, domain_names = prediction_wrapper(model, test_loader, 0, label_name, save_prediction=True)
    for organ in range(len(label_name)):
        mean_dc = np.mean(dsc_table[:, organ])
        std_dc  = np.std(dsc_table[:, organ])
        log.logger.info("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
    log.logger.info("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    log.logger.info("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    cur_dice_val = error_dict['overall_by_domain']
    os.makedirs(f'{visdir}/final', exist_ok=True)
    for key in out_prediction_list:
        plt.axis('off')
        for i in range(len(out_prediction_list[key]["gth"])):
            plt.subplot(1,3,1)
            plt.title("img")
            plt.imshow(out_prediction_list[key]["img"][:,:,i], cmap='gray')
            plt.subplot(1,3,2)
            plt.title("gth")
            plt.imshow((out_prediction_list[key]["gth"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.subplot(1,3,3)
            plt.title("pred")
            plt.imshow((out_prediction_list[key]["pred"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.savefig(f'{visdir}/final/{key}_student{i}.png')
            plt.close()
            plt.axis('off')
            plt.xticks([]) # 去刻度
            plt.imshow(out_prediction_list[key]["img"][:,:,i], cmap='gray')
            plt.savefig(f'{visdir}/final/{key}_student{i}_img.png',bbox_inches="tight", pad_inches = -0.1)
            plt.close()
            plt.axis('off')
            plt.xticks([]) # 去刻度
            plt.imshow((out_prediction_list[key]["gth"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.savefig(f'{visdir}/final/{key}_student{i}_gth.png',bbox_inches="tight", pad_inches = -0.1)
            plt.close()
            plt.axis('off')
            plt.xticks([]) # 去刻度
            plt.imshow((out_prediction_list[key]["pred"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.savefig(f'{visdir}/final/{key}_student{i}_pred.png',bbox_inches="tight", pad_inches = -0.1)
            plt.close()
            
    teacher.load_state_dict(torch.load(os.path.join(ckptdir,f'teacher_best_epoch.pth'), map_location="cpu")['model'], strict=False)
    T_out_prediction_list, T_dsc_table, T_error_dict, T_domain_names = prediction_wrapper(teacher, test_loader, 0, label_name, save_prediction=True)
    for organ in range(len(label_name)):
        T_mean_dc = np.mean(T_dsc_table[:, organ])
        T_std_dc  = np.std(T_dsc_table[:, organ])
        log.logger.info("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], T_mean_dc, T_std_dc))
    log.logger.info("Overall std dice by sample {:06.5f}".format(T_dsc_table[:, 1:].std()))
    log.logger.info("Overall mean dice by sample {:06.5f}".format( T_dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    T_cur_dice_val = T_error_dict['overall_by_domain']
    for key in T_out_prediction_list:
        plt.axis('off')
        for i in range(len(T_out_prediction_list[key]["gth"])):
            plt.subplot(1,3,1)
            plt.title("img")
            plt.imshow(T_out_prediction_list[key]["img"][:,:,i], cmap='gray')
            plt.subplot(1,3,2)
            plt.title("gth")
            plt.imshow((T_out_prediction_list[key]["gth"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.subplot(1,3,3)
            plt.title("pred")
            plt.imshow((T_out_prediction_list[key]["pred"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.savefig(f'{visdir}/final/{key}_teacher{i}.png')
            plt.close()
            plt.axis('off')
            plt.xticks([]) # 去刻度
            plt.imshow(T_out_prediction_list[key]["img"][:,:,i], cmap='gray')
            plt.savefig(f'{visdir}/final/{key}_teacher{i}_img.png',bbox_inches="tight", pad_inches = -0.1)
            plt.close()
            plt.axis('off')
            plt.xticks([]) # 去刻度
            plt.imshow((T_out_prediction_list[key]["gth"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.savefig(f'{visdir}/final/{key}_teacher{i}_gth.png',bbox_inches="tight", pad_inches = -0.1)
            plt.close()
            plt.axis('off')
            plt.xticks([]) # 去刻度
            plt.imshow((T_out_prediction_list[key]["pred"]).detach().cpu().numpy()[i], vmin=0, vmax=4)
            plt.savefig(f'{visdir}/final/{key}_teacher{i}_pred.png',bbox_inches="tight", pad_inches = -0.1)
            plt.close()
    # imshow_expand(model, criterion,train_loader,test_loader,torch.device('cuda'), 0, cur_iter,SBF_config,config, IIA_model)        