import os
import cv2
from tqdm import tqdm
from metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


def eval_results(preds_data_root,dataset_list):
# preds_data_root = "/media/store/yc/projects/COD/Net3/res/28_wELoss/final"
    f = open(preds_data_root+"log.txt","a+",encoding="UTF-8")
    for _data_name in dataset_list:
        if os.path.exists(preds_data_root+_data_name) == False:
            continue

        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        mae = MAE()
        masks_data_root = "../../Dataset/TestDataset/" +_data_name         # 1

        mask_root = os.path.join(masks_data_root, "GT")
        # pred_root = os.path.join(preds_data_root,_data_name,"preds")           # 2
        pred_root = os.path.join(preds_data_root,_data_name)
        mask_name_list = sorted(os.listdir(mask_root))
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or pred is None:
                continue
            if pred.size != mask.size:
                pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            mae.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = mae.get_results()["mae"]

        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        print(results)
        f.write(_data_name+"\n")
        f.write(str(results) + "\n\n")
        f.flush()
    f.close()

if __name__ == '__main__':
    print("start")
    DATASETS = ['COD10K']
    preds_data_root = "./res/prediction_RDVP_MSD/"
    eval_results(preds_data_root,DATASETS)
    print("end")

