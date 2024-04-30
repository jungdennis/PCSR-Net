import os
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from DLCs.augmentation import pil_augm_lite_v2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Dataset_for_SR_REID(data.Dataset):
    def __init__(self, **kwargs):
        self.database = kwargs['database']
        self.path_hr = kwargs['path_hr']
        self.path_lr = kwargs['path_lr']
        self.path_fold = kwargs['path_fold']
        self.mode = kwargs['mode']
        try:
            self.transform_raw = kwargs['transform']
        except:
            self.transform_raw = transforms.Compose([transforms.ToTensor()])

        if self.mode.split("_")[0] == "train":
            self.path_data = "/train/images"
        elif self.mode.split("_")[0] == "valid":
            self.path_data = "/val/images"
        elif self.mode.split("_")[0] == "test":
            self.path_data = "/test/images"

        self.list_files = os.listdir(self.path_hr + self.path_fold + self.path_data)

        self.label_frame = {}
        self.label_list = []
        for name in self.list_files:
            if self.database == "Reg":
                label = name.split(".")[0].split("_")[4]
                frame = name.split(".")[0].split("_")[3]
            elif self.database == "SYSU":
                label = name.split(".")[0].split("_")[0]
                frame = name.split(".")[0].split("_")[1] + "_" + name.split(".")[0].split("_")[2]

            if label not in self.label_list:
                self.label_frame[label] = []
                self.label_list.append(label)

            self.label_frame[label].append(frame)
            self.label_frame[label].sort()

        if "train" in self.mode or "valid" in self.mode:
            self.img_list = []
            if self.database == "Reg":
                for label in list(self.label_list):
                    _list = self.label_frame[label]
                    for i in range(len(_list)) :
                        img_label = _list[i]
                        name_frag = "_" + img_label + "_" + label
                        for name in self.list_files:
                            if name[-len(name_frag)-4:-4] == name_frag:
                                self.img_list.append(name)
            elif self.database == "SYSU" or self.database == "SYSU_old":
                for label in list(self.label_list):
                    _list = self.label_frame[label]
                    for i in range(len(_list)) :
                        img_frame = _list[i]
                        name_frag = label + "_" + img_frame
                        for name in self.list_files :
                            if name_frag in name :
                                self.img_list.append(name)
            self.img_list = list(set(self.img_list))
        elif "test" in self.mode:
            self.query_list = []
            self.gallery_list = []
            if self.database == "Reg":
                for label in list(self.label_list):
                    _list = self.label_frame[label]
                    center_label = _list[0]
                    name_frag = "_" + center_label + "_" + label
                    for name in self.list_files:
                        if name[-len(name_frag) - 4:-4] == name_frag:
                            self.gallery_list.append(name)

                for label in list(self.label_list):
                    _list = self.label_frame[label]
                    image_label_1 = _list[5]
                    name_frag_1 = image_label_1 + "_" + label
                    image_label_2 = _list[9]
                    name_frag_2 = image_label_2 + "_" + label

                    for name in self.list_files:
                        if name[-len(name_frag_1) - 4:-4] == name_frag_1:
                            self.query_list.append(name)
                        if name[-len(name_frag_2) - 4:-4] == name_frag_2:
                            self.query_list.append(name)
            elif self.database == "SYSU":
                for label in list(self.label_list):
                    _list = self.label_frame[label]
                    center_frame = _list[0]
                    name_frag = label + "_" + center_frame
                    for name in self.list_files:
                        if name_frag in name:
                            self.gallery_list.append(name)

                for label in list(self.label_list):
                    _list = self.label_frame[label]
                    for i in range(1, len(_list)):
                        img_frame = _list[i]
                        name_frag = label + "_" + img_frame
                        for name in self.list_files:
                            if name_frag in name:
                                self.query_list.append(name)

            self.query_list = list(set(self.query_list))
            self.gallery_list = list(set(self.gallery_list))

            if "query" in self.mode:
                self.img_list = self.query_list
            elif "gallery" in self.mode:
                self.img_list = self.gallery_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        _name = self.img_list[idx]

        pil_hr = Image.open(self.path_hr + self.path_fold + self.path_data + "/" + _name)
        pil_lr = Image.open(self.path_lr + self.path_fold + self.path_data + "/" + _name)

        if self.database == "Reg":
            label = _name.split(".")[0].split("_")[4]
        elif self.database == "SYSU" or self.database == "SYSU_old":
            label = _name.split(".")[0].split("_")[0]

        label_index = list(self.label_frame.keys()).index(label)
        label_tensor = torch.Tensor([label_index]).to(torch.int64)

        if "train" in self.mode:
            pil_hr_patch, pil_lr_patch, aug_info = pil_augm_lite_v2(pil_hr,
                                                                    image_pil_add=pil_lr,
                                                                    mode='augment',
                                                                    flip_hori=True,
                                                                    flip_vert=True,
                                                                    return_info=True)

            return self.transform_raw(pil_hr_patch), self.transform_raw(pil_lr_patch), label_tensor, aug_info, _name

        elif "valid" in self.mode:
            return self.transform_raw(pil_hr), self.transform_raw(pil_lr), label_tensor, _name
        elif "test" in self.mode:
            return self.transform_raw(pil_hr), self.transform_raw(pil_lr), _name, label_tensor