import os
import random
import argparse

import torch
import torchvision.transforms as transforms
import ignite
from pytorch_metric_learning import losses
from DLCs.mp_dataloader import DataLoader_multi_worker_FIX
from DLCs.metric_tools import metric_histogram, calc_FAR_FRR_v2, calc_EER
from torchmetrics.classification import AveragePrecision, Accuracy

import numpy as np

from dataset import Dataset_for_SR_REID as Dataset
from model import create_model_sr, create_model_reid

# random seed 고정
SEED = 485
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.cuda.manual_seed(SEED)

# Argparse Setting
parser = argparse.ArgumentParser(description = "Test process of PCSR-Net")

parser.add_argument("--base", required = False)
parser.add_argument('--database', required = False, choices = ["Reg", "SYSU"])
parser.add_argument('--fold', required = False)
parser.add_argument('--scale', required = False, type = int)
parser.add_argument('--noise', required = False, type = int)
parser.add_argument('--lr', required = False, type = float)
parser.add_argument('--decay', required = False, type = float)
parser.add_argument('--lamda', required = False, type = float)
parser.add_argument('--lamdapart', required = False, type = float)
parser.add_argument('--mode', required = False, choices = ["last", "loss", "psnr", "ssim", "all"])
parser.add_argument('--metric', required = False, choices = ["mAP", "rank1", "rank10", 'rank20', "all"])

args = parser.parse_args()

# Mode Setting
BASE = args.base
DATABASE = args.database
FOLD = args.fold
SCALE_FACTOR = args.scale
NOISE = args.noise
LR = args.lr
DECAY = args.decay
LAMDA = args.lamda
LAMDA_PART = args.lamdapart
MODE = args.mode
METRIC = args.metric

path_device= "C:/Users/syjung/Desktop/PCSR-Net"

if DATABASE == "Reg":
    path_img = path_device + "/data/Reg/"
elif DATABASE == "SYSU":
    path_img = path_device + "/data/SYSU/"

path_hr = path_img + "HR"
path_lr = path_img + f"LR_{SCALE_FACTOR}_noise{NOISE}"
path_sr = path_img + f"SR_PCSR/{LR}_{DECAY}_{LAMDA}_{LAMDA_PART}/{BASE}"

path_train = "/train/images"
path_valid = "/val/images"
path_test = "/test/images"

path_log = path_device + f"/log/{LR}_{DECAY}_{LAMDA}_{LAMDA_PART}/{BASE}/{DATABASE}/{FOLD}_set"

transform_raw = transforms.Compose([transforms.ToTensor()])

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_scaler_sr = torch.cuda.amp.GradScaler(enabled=True)
    amp_scaler_reid = torch.cuda.amp.GradScaler(enabled=True)

    model_sr = create_model_sr(scale=SCALE_FACTOR)
    model_sr.to(device)

    model_reid = create_model_reid(DATABASE, BASE, FOLD)
    model_reid.to(device)

    cal_dist = torch.nn.PairwiseDistance(p=2).to(device)

    criterion_sr = torch.nn.L1Loss(reduction='none').to(device)
    criterion_reid = losses.ContrastiveLoss().to(device)

    # PSNR, SSIM
    def ignite_eval_step(engine, batch):
        return batch

    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')

    # Dataset/Dataloader
    dataset_test_query = Dataset(database=DATABASE, path_hr=path_hr, path_lr=path_lr, path_fold=f"/{FOLD}_set",
                                   mode="test_query")
    dataset_test_gallery = Dataset(database=DATABASE, path_hr=path_hr, path_lr=path_lr, path_fold=f"/{FOLD}_set",
                                     mode="test_gallery")
    dataloader_test_query = DataLoader_multi_worker_FIX(dataset=dataset_test_query,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        prefetch_factor=2,
                                                        drop_last=False)
    dataloader_test_gallery = DataLoader_multi_worker_FIX(dataset=dataset_test_gallery,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=4,
                                                          prefetch_factor=2,
                                                          drop_last=False)

    len_gallery = len(dataset_test_gallery.img_list)

    print(f"[Info]")
    print(f"LR Image Option : Scale x{SCALE_FACTOR}, Noise {NOISE}")
    print(f"Train Option : LR {LR}, Decay : {DECAY}, SR Lamda : {LAMDA_PART}, REID Lamda : {LAMDA}")
    print(f"Fold {FOLD} : Gallery {len(dataset_test_gallery.img_list)} / Query {len(dataset_test_query.img_list)}")
    print(f"Metric : {METRIC}")

    if DATABASE == "Reg":
        den = 1000
    else :
        den = 100000

    # Distance 측정
    n_img_A = len(dataset_test_query.img_list) * len(dataset_test_gallery.img_list)
    print(f"Fold {FOLD} Distance Measure")
    checkpoint_path = path_log + "/checkpoint"
    ckpt_list = os.listdir(checkpoint_path)
    ckpt_list = sorted(ckpt_list)

    mode_list = []
    if MODE == "last":
        for ckpt in ckpt_list:
            if not "best" in ckpt:
                mode_list.append(ckpt)
    elif MODE == "loss":
        for ckpt in ckpt_list:
            if "best_loss" in ckpt:
                mode_list.append(ckpt)
    elif MODE == "psnr":
        for ckpt in ckpt_list:
            if "best_psnr" in ckpt:
                mode_list.append(ckpt)
    elif MODE == "ssim":
        for ckpt in ckpt_list:
            if "best_ssim" in ckpt:
                mode_list.append(ckpt)
    elif MODE == "all":
        mode_list = ckpt_list

    mode_list = sorted(mode_list)
    if MODE == "last":
        _ = mode_list[-1]
        mode_list = _

    for mode in mode_list:
        ckpt = torch.load(checkpoint_path + f"/{mode}")
        model_sr.load_state_dict(ckpt["model_sr"])

        print(f"{mode} loaded")

        psnr_test = []
        ssim_test = []
        distance = []
        label = []
        distance_genuine = []
        distance_imposter = []

        model_sr.eval()
        model_reid.eval()
        cnt = 0
        gallery_list = {}
        for dataloader_g in dataloader_test_gallery:
            gallery_hr, gallery_lr, name_g, label_g = dataloader_g
            gallery_hr = gallery_hr.to(device)
            gallery_lr = gallery_lr.to(device)
            label_g = label_g.to(device)

            with torch.no_grad():
                gallery_sr = model_sr(gallery_lr)

                # Calculate PSNR, SSIM
                ts_hr = gallery_hr[0].to(device)
                ts_sr = gallery_sr[0].to(device)

                ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0),
                                                       torch.unsqueeze(ts_hr, 0)
                                                       ]])

                _psnr_test = ignite_result.metrics['psnr']
                _ssim_test = ignite_result.metrics['ssim']

                psnr_test.append(_psnr_test)
                ssim_test.append(_ssim_test)

                feature_gallery, _h, _u, _l = model_reid(gallery_sr)
                _g = label_g.item()
                gallery_list[_g] = feature_gallery

        gallery_key = list(gallery_list.keys())
        gallery_key.sort()

        for dataloader_q in dataloader_test_query:
            query_hr, query_lr, name_q, label_q = dataloader_q
            query_hr = query_hr.to(device)
            query_lr = query_lr.to(device)
            labal_q = label_q.to(device)
            _q = label_q.item()

            with torch.no_grad():
                query_sr = model_sr(query_lr)

                # Calculate PSNR, SSIM
                ts_hr = query_hr[0].to(device)
                ts_sr = query_sr[0].to(device)

                ignite_result = ignite_evaluator.run([[torch.unsqueeze(ts_sr, 0),
                                                       torch.unsqueeze(ts_hr, 0)
                                                       ]])

                _psnr_test = ignite_result.metrics['psnr']
                _ssim_test = ignite_result.metrics['ssim']

                psnr_test.append(_psnr_test)
                ssim_test.append(_ssim_test)

                feature_query, _h, _u, _l = model_reid(gallery_sr)

                distance_list = []
                for _g in gallery_key:
                    feature_gallery = gallery_list[_g]
                    _distance = cal_dist(feature_gallery, feature_query)
                    distance_list.append(1 / _distance.item())

                    if _q == _g:
                        distance_genuine.append(_distance.item())
                    else:
                        distance_imposter.append(_distance.item())

                    cnt += 1
                    if cnt % den == 0:
                        print(f"Calculating Distance... [{cnt}/{n_img_A}]")
                distnace_list = torch.tensor(distance_list).to(device)
                distnace_list = torch.softmax(distnace_list, dim=-1)
                distance.append(distance_list)
                label.append(_q)

        distance = torch.tensor(distance).to(device)
        label = torch.tensor(label).to(device)
        distance_genuine.sort()
        distance_imposter.sort()
        g_A = np.array(distance_genuine)
        i_A = np.array(distance_imposter)
        print(distance.shape, label.shape)

        psnr = sum(psnr_test) / len(psnr_test)
        ssim = sum(ssim_test) / len(ssim_test)

        if METRIC == "all":
            metric_histogram(g_A, i_A, density=True,
                             title=f"Distribution of Distance ({DATABASE}_Fold {FOLD}_{MODE})",
                             save_path=path_log + f"hist_{DATABASE}_{FOLD}_{MODE}.png")

            threshold, FAR, FRR = calc_FAR_FRR_v2(g_A, i_A)
            EER, th = calc_EER(threshold, FAR, FRR)

            mAP_cal = AveragePrecision(task="multiclass", num_classes=len_gallery, average='macro').to(device)
            rank1_cal = Accuracy(task="multiclass", num_classes=len_gallery, average='macro', top_k=1).to(device)
            rank10_cal = Accuracy(task="multiclass", num_classes=len_gallery, average='macro', top_k=10).to(device)
            rank20_cal = Accuracy(task="multiclass", num_classes=len_gallery, average='macro', top_k=20).to(device)

            mAP = mAP_cal(distance, label).item()
            rank1 = rank1_cal(distance, label).item()
            rank10 = rank10_cal(distance, label).item()
            rank20 = rank20_cal(distance, label).item()

            print(f"Fold {FOLD}")
            print(f"PSNR : {psnr}, SSIM : {ssim}")
            print(f"EER : {EER}, mAP : {mAP}, rank-1 : {rank1}, rank-10 : {rank10}, rank-20 : {rank20}")
        else:
            if METRIC == "EER":
                metric_histogram(g_A, i_A, density=True,
                                 title=f"Distribution of Distance ({DATABASE}_Fold {FOLD}_{MODE})",
                                 save_path=path_log + f"hist_{DATABASE}_{FOLD}_{MODE}.png")

                threshold, FAR, FRR = calc_FAR_FRR_v2(g_A, i_A)
                EER, th = calc_EER(threshold, FAR, FRR)
                metric = EER
            else:
                if METRIC == "mAP":
                    metric_cal = AveragePrecision(task="multiclass", num_classes=len_gallery, average='macro').to(device)
                elif "rank" in METRIC:
                    k = int(METRIC.split("k")[-1])
                    metric_cal = Accuracy(task="multiclass", num_classes=len_gallery, average='macro', top_k=k).to(device)
                metric = metric_cal(distance, label).item()

            print(f"Fold {FOLD}")
            print(f"PSNR : {psnr}, SSIM : {ssim}")
            print(f"Fold {FOLD} {METRIC} : {metric}")