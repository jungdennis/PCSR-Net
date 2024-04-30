import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def metric_histogram(list_correct, list_wrong, xlim=None, density=False, title="Distribution of Distance", save_path = None):
    plt.clf()
    plt.hist(list_correct, histtype="step", color="b", bins = int(math.ceil(max(list_wrong)) * 2), density=density)
    plt.hist(list_wrong, histtype="step", color="g", bins = int(math.ceil(max(list_wrong)) * 2), density=density)
    plt.title(title)
    plt.xlabel("Distance")
    if density is False :
        plt.ylabel("Count")
        plt.yscale("log")
    else :
        plt.ylabel("Probability")
    plt.legend(["genuine", "imposter"])
    if xlim is not None:
        plt.xlim(xlim)
    else :
        # plt.xlim([0, 45])
        # plt.xlim([0, 250])
        plt.xlim([min(list_correct), max(list_wrong)])

    # plt.ylim([0, 0.05])

    if save_path is not None :
        plt.savefig(save_path)
    else :
        plt.show()


def calc_metric(list_correct, list_wrong, range=None, save=None):
    if range is None:
        threshold = np.arange(0, max(list_wrong), 0.00001)
    else:
        threshold = np.arange(range[0], range[1], range[2])

    total_positive = len(list_correct)
    total_negative = len(list_wrong)

    list_correct = np.array(list_correct)
    list_wrong = np.array(list_wrong)

    FAR = []
    FRR = []

    precision = []
    recall = []

    cnt = 0
    for th in threshold:
        if cnt % 1000 == 0:
            print(f"Calculating metric... {100 * cnt / len(threshold)}%")

        TP = len(list_correct[list_correct < th])
        FP = len(list_correct[list_correct >= th])      # FP = FA
        TN = len(list_wrong[list_wrong >= th])
        FN = len(list_wrong[list_wrong < th])           # FN = FR

        FAR.append(FP / total_positive)
        FRR.append(FN / total_negative)

        precision.append(TP / (TP + FP))
        recall.append(TP / (TP + FN))

        cnt += 1
        # cnt_FA = 0
        # cnt_FR = 0

    if save is not None:
        torch.save({'threshold': threshold,
                    'FAR': FAR,
                    "FRR": FRR,
                    'precision' : precision,
                    'recall' : recall}, save)

    return threshold, FAR, FRR, precision, recall

def calc_FAR_FRR(list_correct, list_wrong, range = None, save = None) :
    if range is None:
        threshold = np.arange(0, max(list_wrong), 0.00001)
    else:
        threshold = np.arange(range[0], range[1], range[2])

    total_FA = len(list_correct)
    total_FR = len(list_wrong)

    list_correct = np.array(list_correct)
    list_wrong = np.array(list_wrong)

    FAR = []
    FRR = []

    cnt = 0
    for th in threshold :
        if cnt % 1000 == 0 :
            print(f"Calculating FAR/FFR... {100 * cnt/len(threshold)}%")

        cnt_FA = len(list_correct[list_correct >= th])
        cnt_FR = len(list_wrong[list_wrong < th])

        FAR.append(cnt_FA / total_FA)
        FRR.append(cnt_FR / total_FR)

        cnt += 1
        # cnt_FA = 0
        # cnt_FR = 0
        
    if save is not None :
        torch.save({'threshold': threshold,
            'FAR': FAR,
            "FRR" : FRR}, save)
            
    return threshold, FAR, FRR

def calc_FAR_FRR_v2(list_genuine, list_imposter, show_log = True, range=None, save=None):
    if range is None:
        threshold = np.arange(0, max(list_imposter) + 0.0001, 0.00001)
        # threshold = np.arange(0, max(list_imposter) + 0.0001, 0.0000001)
    else:
        threshold = np.arange(range[0], range[1] + 0.0001, 0.00001)

    list_genuine = np.array(list_genuine)
    list_imposter = np.array(list_imposter)

    # hist_dict = {}
    # for th in threshold :
    #     if int(th * 100000) % 1000 == 0:
    #         print(f"Calculating hist... {100 * th / max(list_imposter)}%")
    #     cnt_genu = len(list_genuine[list_genuine == th])
    #     cnt_impo = len(list_imposter[list_imposter == th])
    #
    #     hist_dict[th] = [cnt_genu, cnt_impo]

    genuine_dict = {}
    _cnt_g = 0
    for distance in list_genuine:
        distance_round = float(int(distance * (10 ** 5)) / (10 ** 5))
        # distance_round = float(int(distance * (10 ** 7)) / (10 ** 7))
        try:
            genuine_dict[distance_round] += 1
        except:
            genuine_dict[distance_round] = 1

        _cnt_g += 1

    imposter_dict = {}
    _cnt_i = 0
    for distance in list_imposter:
        distance_round = float(int(distance * (10 ** 5)) / (10 ** 5))
        # distance_round = float(int(distance * (10 ** 7)) / (10 ** 7))
        try:
            imposter_dict[distance_round] += 1
        except:
            imposter_dict[distance_round] = 1

        _cnt_i += 1

    total_FA = len(list_genuine)
    total_FR = len(list_imposter)

    FAR = []
    FRR = []

    cnt = 0

    # print(genuine_dict)
    # print(imposter_dict)

    key_genuine = list(genuine_dict.keys())
    key_imposter = list(imposter_dict.keys())

    initial_flag_1 = True
    initial_flag_2 = True
    for th in threshold:
        th = float(int(th * (10 ** 5)) / (10 ** 5))
        # th = float(int(th * (10 ** 7)) / (10 ** 7))
        '''
        cnt_FA = len(list_correct[list_correct >= th])
        cnt_FR = len(list_wrong[list_wrong < th])
        '''
        if initial_flag_1 :
            try :
                cnt_FA = total_FA - genuine_dict[th]
                cnt_genu = 0
            except :
                cnt_FA = total_FA
                cnt_genu = 0
            initial_flag_1 = False
        else :
            cnt_FA = cnt_FA - cnt_genu
            try :
                cnt_genu = genuine_dict[th]
            except :
                cnt_genu = 0

        try :
            FAR.append(cnt_FA / total_FA)
        except ZeroDivisionError :
            FAR.append(1)

        if initial_flag_2 :
            try :
                cnt_FR = imposter_dict[th]
                cnt_impo = 0
            except :
                cnt_FR = 0
                cnt_impo = 0
            initial_flag_2 = False
        else :
            cnt_FR = cnt_FR + cnt_impo
            try :
                cnt_impo = imposter_dict[th]
            except :
                cnt_impo = 0

        try :
            FRR.append(cnt_FR / total_FR)
        except ZeroDivisionError :
            FRR.append(0)

        if show_log and cnt % 10000 == 0 :
            print(f"Distance {th} : FA {cnt_FA}, FR {cnt_FR}")

        cnt += 1
    if show_log :
        print(f"total : FA {total_FA}, FR {total_FR}")

    if save is not None:
        torch.save({'threshold': threshold,
                    'FAR': FAR,
                    "FRR": FRR}, save)

    return threshold, FAR, FRR

def calc_FAR_FRR_v3(list_genuine, list_imposter, show_log = True, range=None, save=None):
    if range is None:
        # threshold = np.arange(min(0, min(list_genuine)), max(list_imposter) + 0.0001, 0.00000001)
        threshold = np.arange(min(0, min(list_genuine)), max(list_imposter) + 0.0001, 0.00001)
    else:
        threshold = np.arange(range[0], range[1] + 0.0001, 0.00001)

    list_genuine = np.array(list_genuine)
    list_imposter = np.array(list_imposter)

    # hist_dict = {}
    # for th in threshold :
    #     if int(th * 100000) % 1000 == 0:
    #         print(f"Calculating hist... {100 * th / max(list_imposter)}%")
    #     cnt_genu = len(list_genuine[list_genuine == th])
    #     cnt_impo = len(list_imposter[list_imposter == th])
    #
    #     hist_dict[th] = [cnt_genu, cnt_impo]

    genuine_dict = {}
    _cnt_g = 0
    for distance in list_genuine:
        distance_round = float(int(distance * (10 ** 5)) / (10 ** 5))
        try:
            genuine_dict[distance_round] += 1
        except:
            genuine_dict[distance_round] = 1

        _cnt_g += 1

    imposter_dict = {}
    _cnt_i = 0
    for distance in list_imposter:
        distance_round = float(int(distance * (10 ** 5)) / (10 ** 5))
        try:
            imposter_dict[distance_round] += 1
        except:
            imposter_dict[distance_round] = 1

        _cnt_i += 1

    total_FA = len(list_genuine)
    total_FR = len(list_imposter)

    FAR = []
    FRR = []
    precision = []
    recall = []

    cnt = 0

    # print(genuine_dict)
    # print(imposter_dict)

    key_genuine = list(genuine_dict.keys())
    key_imposter = list(imposter_dict.keys())

    initial_flag_1 = True
    initial_flag_2 = True
    for th in threshold:
        th = float(int(th * (10 ** 5)) / (10 ** 5))
        '''
        cnt_FA = len(list_correct[list_correct >= th])
        cnt_FR = len(list_wrong[list_wrong < th])
        cnt_TA = len(list_correct[list_correct < th])
        cnt_TR = len(list_wrong[list_wrong >= th])
        '''

        if initial_flag_1 :
            try :
                cnt_FA = total_FA - genuine_dict[th]
                cnt_TA = total_FA - cnt_FA
                cnt_genu = 0
            except :
                cnt_FA = total_FA
                cnt_TA = total_FA - cnt_FA
                cnt_genu = 0
            initial_flag_1 = False
        else :
            cnt_FA = cnt_FA - cnt_genu
            cnt_TA = total_FA - cnt_FA
            try :
                cnt_genu = genuine_dict[th]
            except :
                cnt_genu = 0

        try :
            FAR.append(cnt_FA / total_FA)
        except ZeroDivisionError :
            FAR.append(1)

        if initial_flag_2 :
            try :
                cnt_FR = imposter_dict[th]
                cnt_TR = total_FR - cnt_FR
                cnt_impo = 0
            except :
                cnt_FR = 0
                cnt_TR = total_FR - cnt_FR
                cnt_impo = 0
            initial_flag_2 = False
        else :
            cnt_FR = cnt_FR + cnt_impo
            cnt_TR = total_FR - cnt_FR
            try :
                cnt_impo = imposter_dict[th]
            except :
                cnt_impo = 0

        try :
            FRR.append(cnt_FR / total_FR)
        except ZeroDivisionError :
            FRR.append(0)

        try:
            recall.append(cnt_TA / (cnt_TA + cnt_FA))
        except:
            recall.append(0)

        try:
            precision.append(cnt_TA / (cnt_TA + cnt_FR))
        except:
            precision.append(1)

        if show_log and cnt % 10000 == 0 :
            print(f"Distance {th} : TA {cnt_TA}, FA {cnt_FA}, TR {cnt_TR}, FR {cnt_FR}")

        cnt += 1
    if show_log :
        print(f"total : TR {total_FA}, FA 0, TR 0, FR {total_FR}")

    if save is not None:
        torch.save({'threshold': threshold,
                    'FAR': FAR,
                    "FRR": FRR,
                    "precision" : precision,
                    'recall' : recall}, save)

    return threshold, FAR, FRR, precision, recall


def calc_EER(threshold, FAR, FRR, print_log = False, for_graph = False, save = None) :
    _x = []
    for i in range(len(threshold)) :
        if FAR[i] > FRR[i] :
            pass
        elif FAR[i] == FRR[i] :
            EER = FAR[i]
            th = threshold[i]
            break
        elif FAR[i] < FRR[i] :
            if print_log :
                print(f"i : {threshold[i]}, {FAR[i]}, {FRR[i]}")
                print(f"i-1 : {threshold[i-1]}, {FAR[i-1]}, {FRR[i-1]}")

            num = (threshold[i-1]-threshold[i])*(FRR[i-1]-FRR[i])-(threshold[i-1]-threshold[i])*(FAR[i-1]-FAR[i])
            den_EER = (threshold[i-1]*FAR[i]-threshold[i]*FAR[i-1])*(FRR[i-1]-FRR[i])-(threshold[i-1]*FRR[i]-threshold[i]*FRR[i-1])*(FAR[i-1]-FAR[i])
            den_th = (threshold[i-1]*FAR[i]-threshold[i]*FAR[i-1])*(threshold[i-1]-threshold[i])-(threshold[i-1]*FRR[i]-threshold[i]*FRR[i-1])*(threshold[i-1]-threshold[i])
            EER = den_EER / num
            th = den_th / num
            break

    if for_graph is True :
        print(f"EER : {EER}, threshold : {th}")
    if save is not None:
        torch.save({"EER" : EER, "th" : th}, save)

    return EER, th

def graph_FAR_FRR(threshold, FAR, FRR, title=None, show_EER = None,
                  xlim = None, ylim = None, log = False, save_path = None) :
    plt.clf()
    plt.plot(threshold, FAR, color="b")
    plt.plot(threshold, FRR, color="g")
    if show_EER :
        y, x = calc_EER(threshold, FAR, FRR, print_log = False, for_graph = False, save = None)
        plt.scatter(x, y, marker ="o", color = "r")
        
    if title is not None:
        plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Rate")
    if log :
        plt.yscale("log")

    plt.legend(["FAR", "FRR", "EER"])

    if xlim is not None :
        plt.xlim(xlim)
    if ylim is not None :
        plt.ylim(xlim)

    if save_path is not None :
        plt.savefig(save_path)
    else :
        plt.show()

def graph_ROC(FAR, FRR, EER = None, cross = False, title = "ROC Curve", save_path = None) :
    plt.clf()
    plt.plot(FAR, FRR, color="b")
    # if EER is not None :
    #     plt.scatter(EER, EER, marker="o", color="r")
    # if cross is True :
    #     x=np.linspace(0.0, 1.0, 1000)
    #     plt.plot(x, x, linestyle = "--", color = "b")
    x = np.linspace(0.0, 1.0, 1000)
    plt.plot(x, x, linestyle = "--", color = "r")
    plt.title(title)
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("False Reject Rate (FRR)")

    plt.xlim([-0.01, 1.0])
    plt.ylim([1.0, -0.01])

    if EER is not None :
        plt.legend(["ROC", "EER line"])

    if save_path is not None :
        plt.savefig(save_path)
    else :
        plt.show()
    
def graph_PR(precision, recall, title = "PR Curve", save_path = None) :
    plt.clf()
    plt.plot(recall, precision, color="b")

    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])

    if save_path is not None :
        plt.savefig(save_path)
    else :
        plt.show()