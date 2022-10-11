from glob import glob
import torch
import cv2
import os
import time
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, save_one_box, scale_coords
from utils.plots import Annotator, colors
from sklearn.metrics import confusion_matrix


def infer(model,dataset):
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to("cuda:0")
        img = img.float()
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]
    # Inference
    result = model(img)[0]
    
    pred = non_max_suppression(result, conf_thres=0.25, iou_thres=0.45, classes=None)

    ######## visualize
    for i, det in enumerate(pred):  # per image
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_dir = "test_result"
        save_path = f"{save_dir}/{p.name}"  # img.jpg
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        save_crop = False
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=1, example=str(model.names))
        names = model.names
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            print(det)
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                c = int(cls)  # integer class
                label = names[c]
                annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                

        # Stream results
            im0 = annotator.result()        
            cv2.imwrite(f"{save_path}",im0)
        # cv2.imshow(str(p), im0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
            return p.name,im0,c

def get_label(label_dir,label_col,img_name):
    label = []
    for col in label_col:
        if img_name.split(".jpg")[0] == col.split(".txt")[0]:
            file = open(os.path.join(label_dir,col),"r+")
            try:
                text = file.readlines()[0]
                # if len(text) > 1:
                #     print(f"multi label, {img_name}")
                #     time.sleep(5)
                label = text.split(" ")[0]
            except:
                pass
            
    return int(label)

if __name__ == '__main__':
    weights = "runs/train/exp/weights/best.pt"
    model = attempt_load(weights, map_location="cuda:0")

    label_dir = "../datasets/coco128/labels/train2017"
    label_col = os.listdir(label_dir)

    prediction = []
    labels = []
    for img in glob("../datasets/coco128/images/train2017" + "/*.jpg"):
        #img1 = 'IMG_Im0058_M19_C00220383-09_FEdge_Defect_U6_E3753_IOISO_UmnU.d_8.png'
        # label = get_label(label_dir,label_col,os.path.basename(img))

        imgsz = check_img_size(128, s=64)
        dataset = LoadImages(img, img_size=imgsz, stride=64, auto=True)
        result = infer(model,dataset)
        if result is not None:
            name,img,pred_cls = result
            # labels.append(label)
            prediction.append(pred_cls)
            # if pred_cls != label:
            #     cv2.imwrite(f"underkill/{name}_{label}.png",img)
            
    # print(prediction)
    # print(labels)
    # print("confusion matrix: ")
    # print(confusion_matrix(labels,prediction))