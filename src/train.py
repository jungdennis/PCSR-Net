import os
import random
import argparse

import torch
import torchvision.transforms as transforms
import ignite
from torchvision.transforms.functional import to_pil_image
from pytorch_metric_learning import losses

from DLCs.augmentation import pil_augm_lite_v2
from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.data_record import RecordBox
from DLCs.sr_tools import graph_loss, graph_single

import numpy as np

from dataset import Dataset_for_SR_REID as Dataset

from model import create_model_sr, create_model_reid

# random seed
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.cuda.manual_seed(SEED)

# Argparse Setting
parser = argparse.ArgumentParser(description = "Train process of PCSR-Net")

parser.add_argument("--base", required = False)
parser.add_argument('--database', required = False)
parser.add_argument('--fold', required = False)
parser.add_argument('--scale', required = False, type = int)
parser.add_argument('--noise', required = False, type = int)
parser.add_argument('--epoch', required = False, type = int)
parser.add_argument('--batch', required = False, type = int)
parser.add_argument('--lr', required = False, type = float)
parser.add_argument('--decay', required = False, type = float)
parser.add_argument('--lamda', required = False, type = float)
parser.add_argument('--lamdapart', required = False, type = float)
parser.add_argument("--load", required = False, action = 'store_true')

args = parser.parse_args()

# Mode Setting
BASE = args.base
DATABASE = args.database
FOLD = args.fold
SCALE_FACTOR = args.scale
NOISE = args.noise
EPOCH = args.epoch
BATCH_SIZE = args.batch
LR = args.lr
DECAY = args.decay
LAMDA = args.lamda
LAMDA_PART = args.lamdapart
LOAD = args.load

# Datapath
path_device= "C:/Users/syjung/Desktop/PCSR-Net"

if DATABASE == "Reg":
    path_img = path_device + "/data/Reg/"
elif DATABASE == "SYSU":
    path_img = path_device + "/data/SYSU/"

path_hr = path_img + "HR"
path_lr = path_img + f"LR_{SCALE_FACTOR}_noise{NOISE}"
path_sr = path_img + f"SR_PCSR/{LR}_{DECAY}_{LAMDA}_{LAMDA_PART}/{BASE}"

path_fold = f"/{FOLD}_set"

path_train = "/train/images"
path_valid = "/val/images"
path_test = "/test/images"

path_log = path_device + f"/log/{LR}_{DECAY}_{LAMDA}_{LAMDA_PART}/{BASE}/{DATABASE}/{FOLD}_set"

transform_raw = transforms.Compose([transforms.ToTensor()])

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def lr_schedule(epoch):
    return 0.9865 ** epoch

def save_model(path_log, i_epoch, model_sr, model_reid, optimizer, scheduler, lr_list,
               loss_train_list, psnr_train_list, ssim_train_list,
               loss_valid_list, psnr_valid_list, ssim_valid_list,
               save_name=None):
    if i_epoch < 10:
        epoch_save = f"00{i_epoch}"
    elif i_epoch < 100:
        epoch_save = f"0{i_epoch}"
    else:
        epoch_save = f"{i_epoch}"

    if save_name is not None:
        name = save_name
    else:
        name = epoch_save

    try:
        torch.save({
            'epoch': i_epoch,
            'model_sr': model_sr.state_dict(),
            'model_reid': model_reid.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'lr_sr': lr_list,
            'loss_train': loss_train_list,
            'psnr_train': psnr_train_list,
            'ssim_train': ssim_train_list,
            'loss_valid': loss_valid_list,
            'psnr_valid': psnr_valid_list,
            'ssim_valid': ssim_valid_list,
        }, path_log + f"/checkpoint/{name}.pt")
    except:
        os.makedirs(path_log + "/checkpoint")
        torch.save({
            'epoch': i_epoch,
            'model_sr': model_sr.state_dict(),
            'model_reid': model_reid.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'lr_sr': lr_list,
            'loss_train': loss_train_list,
            'psnr_train': psnr_train_list,
            'ssim_train': ssim_train_list,
            'loss_valid': loss_valid_list,
            'psnr_valid': psnr_valid_list,
            'ssim_valid': ssim_valid_list,
        }, path_log + f"/checkpoint/{name}.pt")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

    model_sr = create_model_sr(scale=SCALE_FACTOR)
    model_sr.to(device)

    model_reid = create_model_reid(DATABASE, BASE, FOLD)
    model_reid.to(device)

    criterion_sr = torch.nn.L1Loss(reduction='none').to(device)
    criterion_reid = losses.ContrastiveLoss().to(device)

    optimizer = torch.optim.Adam(model_sr.parameters(), lr=LR, weight_decay=DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    # PSNR, SSIM
    def ignite_eval_step(engine, batch):
        return batch

    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')

    lr = RecordBox(name="learning_rate", is_print=False)


    loss_train = RecordBox(name="loss_train", is_print=False)
    psnr_train = RecordBox(name="psnr_train", is_print=False)
    ssim_train = RecordBox(name="ssim_train", is_print=False)

    loss_valid = RecordBox(name="loss_valid", is_print=False)
    psnr_valid = RecordBox(name="psnr_valid", is_print=False)
    ssim_valid = RecordBox(name="ssim_valid", is_print=False)

    lr_list = []

    loss_train_list = []
    psnr_train_list = []
    ssim_train_list = []

    loss_valid_list = []
    psnr_valid_list = []
    ssim_valid_list = []

    # Dataset/Dataloader
    dataset_train = Dataset(database=DATABASE, path_hr=path_hr, path_lr=path_lr, path_fold=path_fold, mode="train")

    dataset_valid = Dataset(database=DATABASE, path_hr=path_hr, path_lr=path_lr, path_fold=path_fold, mode="valid")

    dataloader_train = DataLoader_multi_worker_FIX(dataset=dataset_train,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=False)
    dataloader_valid = DataLoader_multi_worker_FIX(dataset=dataset_valid,
                                                   batch_size=3,
                                                   shuffle=False,
                                                   num_workers=2,
                                                   prefetch_factor=2,
                                                   drop_last=True)

    if LOAD:
        ckpt_path = path_log + "/checkpoint"
        ckpt_list = os.listdir(ckpt_path)
        ckpt_list = sorted(ckpt_list)
        ckpt = torch.load(ckpt_path + f"/{ckpt_list[-1]}")

        model_sr.load_state_dict(ckpt["model_sr"])
        model_reid.load_state_dict(ckpt["model_reid"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        loss_train_list = ckpt["loss_train"]
        psnr_train_list = ckpt["psnr_train"]
        ssim_train_list = ckpt["ssim_train"]

        loss_valid_list = ckpt["loss_valid"]
        psnr_valid_list = ckpt["psnr_valid"]
        ssim_valid_list = ckpt["ssim_valid"]

        try:
            lr_list = ckpt["lr"]
        except:
            pass

        i_epoch = ckpt["epoch"]

        del ckpt

    else:
        i_epoch = 0

    print(f"[Fold {FOLD} Info (Train/Valid)]")
    print(f"LR Image Option : Scale x{SCALE_FACTOR}, Noise {NOISE}")
    print(f"Train Option : Epoch {EPOCH}, LR {LR}, Decay : {DECAY}, SR Lamda : {LAMDA_PART}, REID Lamda : {LAMDA}")
    print(f"Number of image : Train {len(dataset_train.img_list)}, Valid {len(dataset_valid.img_list)}")

    # train, valid
    size = len(dataloader_train.dataset)

    best_loss = 99
    best_psnr = 0
    best_ssim = 0

    for i_epoch_raw in range(EPOCH):
        i_epoch += 1

        if i_epoch > EPOCH:
            break

        print(f"<Epoch {i_epoch}>")

        # train
        optimizer.zero_grad()
        model_sr.train()
        model_reid.train()
        for p in model_reid.model_base.parameters():
            p.requires_grad = False
        for p in model_reid.part_attention.parameters():
            p.requires_grad = True

        for batch, i_dataloader in enumerate(dataloader_train):
            i_batch_hr, i_batch_lr, i_batch_label, i_aug_info, i_batch_name = i_dataloader
            i_batch_hr = i_batch_hr.to(device)
            i_batch_lr = i_batch_lr.to(device)
            i_batch_label = i_batch_label.to(device)
            i_batch_label = i_batch_label.squeeze()

            i_batch_lr = i_batch_lr.requires_grad_(True)

            # SR Network
            i_batch_sr = model_sr(i_batch_lr)

            _loss_train_whole = criterion_sr(i_batch_sr, i_batch_hr).mean()

            hr_head, hr_upper, hr_lower = i_batch_hr[:, :, :51, :], i_batch_hr[:, :, 51:153, :], i_batch_hr[:, :, 153:, :]
            sr_head, sr_upper, sr_lower = i_batch_sr[:, :, :51, :], i_batch_sr[:, :, 51:153, :], i_batch_sr[:, :, 153:, :]

            _loss_train_head = criterion_sr(sr_head, hr_head).mean([-3, -2, -1]).unsqueeze(-1)
            _loss_train_upper = criterion_sr(sr_upper, hr_upper).mean([-3, -2, -1]).unsqueeze(-1)
            _loss_train_lower = criterion_sr(sr_lower, hr_lower).mean([-3, -2, -1]).unsqueeze(-1)

            with torch.no_grad():
                sr_reverse = []
                for i_batch in range(len(i_batch_sr)):
                    # Calculate PSNR, SSIM
                    ts_hr = i_batch_hr[i_batch].to(device)
                    ts_sr = i_batch_sr[i_batch].to(device)

                    ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0),
                                                           torch.unsqueeze(ts_hr, 0)
                                                           ]])

                    _psnr_train = ignite_result.metrics['psnr']
                    _ssim_train = ignite_result.metrics['ssim']
                    psnr_train.add_item(_psnr_train)
                    ssim_train.add_item(_ssim_train)

                    ts_hr = torch.clamp(ts_hr, min=0, max=1).to(device)
                    ts_sr = torch.clamp(ts_sr, min=0, max=1).to(device)  # B C H W
                    name = i_batch_name[i_batch]

                    pil_sr = to_pil_image(ts_sr)
                    aug_info = [i_aug_info[0][i_batch].item(), i_aug_info[1][i_batch].item()]
                    pil_sr = pil_augm_lite_v2(pil_sr, mode="reverse", input_info=aug_info)

                    sr_reverse.append(transform_raw(pil_sr))

            # ReID Network
            i_batch_sr_aug = torch.unsqueeze(sr_reverse[0], 0)
            for i in range(1, len(sr_reverse)):
                i_batch_sr_aug = torch.concat((i_batch_sr_aug, torch.unsqueeze(sr_reverse[i], 0)))
            i_batch_sr_aug = i_batch_sr_aug.to(device)
            i_batch_sr_aug = i_batch_sr_aug.requires_grad_(True)

            feature_sr, _h, _u, _l = model_reid(i_batch_sr_aug)
            _loss_train_reid = criterion_reid(feature_sr, i_batch_label)

            _loss_train_head = torch.mul(_loss_train_head, _h)
            _loss_train_upper = torch.mul(_loss_train_upper, _u)
            _loss_train_lower = torch.mul(_loss_train_lower, _l)

            _loss_train_part = _loss_train_head + _loss_train_upper + _loss_train_lower
            _loss_train_part = _loss_train_part.mean()

            _loss_train_sr = _loss_train_whole + LAMDA_PART * _loss_train_part

            _loss_train = _loss_train_sr + LAMDA * _loss_train_reid
            loss_train.add_item(_loss_train.item())

            amp_scaler.scale(_loss_train).backward(retain_graph=False)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()

            # RecordBox
            loss_train.update_batch()
            psnr_train.update_batch()
            ssim_train.update_batch()

            if batch % 20 == 0:
                current = batch * len(i_batch_lr)
                print("Total loss : {:.6f}  [{}/{}]".format(_loss_train.item(), current, size))

        lr.add_item(scheduler.get_last_lr()[0])
        scheduler.step()
        lr.update_batch()

        # valid
        model_sr.eval()
        model_reid.eval()
        for i_dataloader in dataloader_valid:
            i_batch_hr, i_batch_lr, i_batch_label, i_batch_name = i_dataloader
            i_batch_hr = i_batch_hr.to(device)
            i_batch_lr = i_batch_lr.to(device)
            i_batch_label = i_batch_label.squeeze().to(device)

            with torch.no_grad():
                # SR Network
                i_batch_sr = model_sr(i_batch_lr)

                _loss_valid_whole = criterion_sr(i_batch_sr, i_batch_hr).mean()
                err_check = torch.all(torch.isnan(_loss_valid_whole)) or torch.all(torch.isinf(_loss_valid_whole))

                if err_check:
                    i_batch_sr = torch.clamp(i_batch_sr, min=0, max=1).to(device)
                    _loss_valid_whole = criterion_sr(i_batch_sr, i_batch_hr).mean()

                hr_head, hr_upper, hr_lower = i_batch_hr[:, :, :51, :], i_batch_hr[:, :, 51:153, :], i_batch_hr[:,
                                                                                                     :, 153:, :]
                sr_head, sr_upper, sr_lower = i_batch_sr[:, :, :51, :], i_batch_sr[:, :, 51:153, :], i_batch_sr[:,
                                                                                                     :, 153:, :]

                _loss_valid_head = criterion_sr(sr_head, hr_head).mean([-3, -2, -1]).unsqueeze(-1)
                _loss_valid_upper = criterion_sr(sr_upper, hr_upper).mean([-3, -2, -1]).unsqueeze(-1)
                _loss_valid_lower = criterion_sr(sr_lower, hr_lower).mean([-3, -2, -1]).unsqueeze(-1)

                # PSNR, SSIM Calculate
                for i_batch in range(len(i_batch_sr)):
                    ts_hr = i_batch_hr[i_batch].to(device)
                    ts_sr = i_batch_sr[i_batch].to(device)

                    ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0),
                                                           torch.unsqueeze(ts_hr, 0)
                                                           ]])

                    _psnr_valid = ignite_result.metrics['psnr']
                    _ssim_valid = ignite_result.metrics['ssim']
                    psnr_valid.add_item(_psnr_valid)
                    ssim_valid.add_item(_ssim_valid)

                    ts_hr = torch.clamp(ts_hr, min=0, max=1).to(device)
                    ts_sr = torch.clamp(ts_sr, min=0, max=1).to(device)  # B C H W
                    name = i_batch_name[i_batch]

                # ReID Network
                feature_sr, _h, _u, _l = model_reid(i_batch_sr)
                _loss_valid_reid = criterion_reid(feature_sr, i_batch_label)

                _loss_valid_head = torch.mul(_loss_valid_head, _h)
                _loss_valid_upper = torch.mul(_loss_valid_upper, _u)
                _loss_valid_lower = torch.mul(_loss_valid_lower, _l)

                _loss_valid_part = _loss_valid_head + _loss_valid_upper + _loss_valid_lower
                _loss_valid_part = _loss_valid_part.mean()

                _loss_valid_sr = _loss_valid_whole + LAMDA_PART * _loss_valid_part

                _loss_valid = _loss_valid_sr + LAMDA * _loss_valid_reid
                loss_valid.add_item(_loss_valid.item())

            loss_valid.update_batch()
            psnr_valid.update_batch()
            ssim_valid.update_batch()

        _lt = loss_train.update_epoch(is_return=True, path=path_log)
        _pt = psnr_train.update_epoch(is_return=True, path=path_log)
        _st = ssim_train.update_epoch(is_return=True, path=path_log)

        _lv = loss_valid.update_epoch(is_return=True, path=path_log)
        _pv = psnr_valid.update_epoch(is_return=True, path=path_log)
        _sv = ssim_valid.update_epoch(is_return=True, path=path_log)
        _lr = lr.update_epoch(is_return=True, path=path_log)

        lr_list.append(_lr)

        loss_train_list.append(_lt)
        psnr_train_list.append(_pt)
        ssim_train_list.append(_st)

        loss_valid_list.append(_lv)
        psnr_valid_list.append(_pv)
        ssim_valid_list.append(_sv)

        print("train : Loss {:.6f}, PSNR {:.4f}, SSIM : {:.6f}".format(_lt, _pt, _st))
        print("valid : Loss {:.6f}, PSNR {:.4f}, SSIM : {:.6f}".format(_lv, _pv, _sv))

        save_model(path_log, i_epoch, model_sr, model_reid, optimizer, scheduler, lr_list,
                   loss_train_list, psnr_train_list, ssim_train_list,
                   loss_valid_list, psnr_valid_list, ssim_valid_list)

        if _lv <= best_loss:
            save_model(path_log, i_epoch, model_sr, model_reid, optimizer, scheduler, lr_list,
                       loss_train_list, psnr_train_list, ssim_train_list,
                       loss_valid_list, psnr_valid_list, ssim_valid_list,
                       save_name="best_loss")
            best_loss = _lv
            print("Best loss updated!")
        if _pv >= best_psnr:
            save_model(path_log, i_epoch, model_sr, model_reid, optimizer, scheduler, lr_list,
                       loss_train_list, psnr_train_list, ssim_train_list,
                       loss_valid_list, psnr_valid_list, ssim_valid_list,
                       save_name="best_loss")
            best_psnr = _pv
            print("Best psnr updated!")
        if _sv >= best_ssim:
            save_model(path_log, i_epoch, model_sr, model_reid, optimizer, scheduler, lr_list,
                       loss_train_list, psnr_train_list, ssim_train_list,
                       loss_valid_list, psnr_valid_list, ssim_valid_list,
                       save_name="best_loss")
            best_ssim = _sv
            print("Best ssim updated!")

        if i_epoch % 10 == 0:
            graph_loss(loss_train_list, loss_valid_list, save=path_log + "/loss.png", title="Graph of Loss")
            graph_single(lr_list, "lr", save=path_log + "/lr.png", title=f"Graph of LR")

            graph_single(psnr_train_list, "PSNR", save=path_log + f"/psnr_train.png", title=f"Graph of PSNR_train", print_max=True)
            graph_single(psnr_valid_list, "PSNR", save=path_log + f"/psnr_valid.png", title=f"Graph of PSNR_valid", print_max=True)
            graph_single(ssim_train_list, "SSIM", save=path_log + f"/ssim_train.png", title=f"Graph of SSIM_train", print_max=True)
            graph_single(ssim_valid_list, "SSIM", save=path_log + f"/ssim_valid.png", title=f"Graph of SSIM_valid", print_max=True)
        print("------------------------------------------------------------------------")

    graph_loss(loss_train_list, loss_valid_list, save = path_log + "/loss.png", title = "Graph of Loss")
    graph_single(lr_list, "lr", save = path_log + "/lr.png", title = f"Graph of LR")