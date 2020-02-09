import cv2
import torch
import numpy as np 
import os 

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def plot_preds(numpy_img, preds):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale 
    fontScale = 0.7
    # Blue color in BGR 
    color = (0, 0, 255) 
    # Line thickness of 2 px 
    thickness = 2

    boxes = preds['boxes'].cpu().detach().numpy()
    labels = preds['labels']
    # for box in boxes:
    #     y = int(box[1] - 5)
    #     x = box[0]
    #     numpy_img = cv2.rectangle(
    #         numpy_img, 
    #         (box[0],box[1]),
    #         (box[2],box[3]), 
    #         255,
    #         3
    #     ) 
    #     cv2.putText( numpy_img, '000', (x,y), font, fontScale, color, thickness)
    
    for i in range(len(boxes)):
        y = int(boxes[i][1] - 5)
        x = boxes[i][0]
        numpy_img = cv2.rectangle(
            numpy_img, 
            (boxes[i][0],boxes[i][1]),
            (boxes[i][2],boxes[i][3]), 
            255,
            3
        ) 
        cv2.putText( numpy_img, labels[i], (x,y), font, fontScale, color, thickness)

    return numpy_img.get()

class ImageProcessing(object):
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print("Using device:", self.device)
        
    def object_detection(self, image):
        
        image_array = np.asarray(bytearray(image), dtype="uint8")
        img_opencv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        img_opencv_rgb = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        
        img_numpy = img_opencv_rgb[:,:,::-1]

        img = torch.from_numpy(img_numpy.astype('float32')).permute(2,0,1)
        img = (img / 255.).to(self.device)

        predictions = self.model(img[None,...])
        CONF_THRESH = 0.5
        # boxes = predictions[0]['boxes'][(predictions[0]['scores'] > CONF_THRESH) & (predictions[0]['labels'] == 1)]
        # print(predictions[0]['labels'])
        boxes = predictions[0]['boxes'][predictions[0]['scores'] > CONF_THRESH]
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions[0]['labels'].cpu().detach().numpy())]
        # print(pred_class)
        boxes_dict = {}
        boxes_dict['boxes'] = boxes
        boxes_dict['labels'] = pred_class
        
        img_with_boxes = plot_preds(img_numpy, boxes_dict)
        # print(boxes_dict)
        # return boxes_dict['boxes']
        return img_with_boxes.astype('uint')

    