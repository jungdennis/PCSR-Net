import sys
import numpy as np
from PIL import Image

def pil_augm_lite_v2(image_pil, mode, **kwargs):
    # 일반적인 SR 모델 학습법에 사용되는 Random Flip (수평 혹은 수직) + 90' 단위 Rotation 구현
    # 일반적인 경우를 가정하고 작성하였으므로 정사각형 이미지에 대해서만 가동을 보장함

    # v2 : random으로 augmentation 하는 mode와 augmentation을 원복하는 mode 구현

    w, h = image_pil.size

    if w != h:
        print("Error : 입력 이미지가 정사각형이 아닙니다!")
        sys.exit(-9)
    try:
        image_pil_add = kwargs["image_pil_add"]
    except:
        image_pil_add = None

    if mode == "augment":
        flip_hori = kwargs["flip_hori"]
        flip_vert = kwargs["flip_vert"]
        input_info = None
        try:
            return_info = kwargs["return_info"]
        except:
            return_info = True

        if type(flip_hori) is not bool or type(flip_vert) is not bool or type(return_info) is not bool:
            print("Error : flip_hori/flip_vert/return_info는 bool로 입력해주세요.")
            sys.exit(-9)

        aug_option = []

        # Flip 진행
        # 0 : horizontal flip / 1 : vertical flip / 2 : Pass
        if flip_hori == True and flip_vert == True:
            mirror = int(np.random.choice([0, 1, 2]))  # Hori / Vert / Pass
        elif flip_hori == True:
            mirror = int(np.random.choice([0, 2]))  # Hori / Pass
        elif flip_vert == True:
            mirror = int(np.random.choice([1, 2]))  # Vert / Pass
        else:
            mirror = 2  # Pass

        if mirror == 0:
            # FLIP: horizontal (좌우 반전)
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if image_pil_add is not None:
                image_pil_add = image_pil_add.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == 1:
            # FLIP: vertical (상하 반전)
            image_pil = image_pil.transpose(Image.FLIP_TOP_BOTTOM)
            if image_pil_add is not None:
                image_pil_add = image_pil_add.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            # FLIP: Pass
            pass

        aug_option.append(mirror)

        # Rotate 진행 (0, 90, 180, 270)
        degree = int(np.random.choice([0, 90, 180, 270]))

        if degree == 0:
            pass
        else:
            image_pil = image_pil.rotate(degree)
            if image_pil_add is not None:
                image_pil_add = image_pil_add.rotate(degree)

        aug_option.append(degree)

        if return_info == True:
            if image_pil_add is not None:
                return image_pil, image_pil_add, aug_option
            else:
                return image_pil, aug_option
        else:
            if image_pil_add is not None:
                return image_pil, image_pil_add
            else:
                return image_pil


    elif mode == "reverse":
        flip_hori = None
        flip_vert = None
        input_info = kwargs["input_info"]
        return_info = False

        if type(input_info) is not list:
            print("Error : 정확한 augmentation option을 입력해주세요.")
            sys.exit(-9)

        mirror, degree = input_info

        # Rotate 진행 (0, 90, 180, 270)
        degree = 360 - degree

        if degree == 360:
            pass
        else:
            image_pil = image_pil.rotate(degree)
            if image_pil_add is not None:
                image_pil_add = image_pil_add.rotate(degree)

        # Flip 진행
        if mirror == 0:
            # FLIP: horizontal (좌우 반전)
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if image_pil_add is not None:
                image_pil_add = image_pil_add.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == 1:
            # FLIP: vertical (상하 반전)
            image_pil = image_pil.transpose(Image.FLIP_TOP_BOTTOM)
            if image_pil_add is not None:
                image_pil_add = image_pil_add.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            # FLIP: Pass
            pass

        if image_pil_add is not None:
            return image_pil, image_pil_add
        else:
            return image_pil

    else:
        print("Error : augment/reverse 모드만 지원합니다.")
        sys.exit(-9)