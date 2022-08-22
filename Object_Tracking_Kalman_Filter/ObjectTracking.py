import os
#from util import bounding
from sort import Sort
import numpy as np
import torch
import cv2
import shutil


VIDEO_FILE = "./video.mp4"
DETECT = "person"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
vidcap = cv2.VideoCapture(VIDEO_FILE)
fps = vidcap.get(cv2.CAP_PROP_FPS)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
folder_out = "Track"
if os.path.exists(folder_out):
    shutil.rmtree(folder_out)
os.makedirs(folder_out)

draw_imgs = []

sort = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

i = 0
while True:
    ret, frame = vidcap.read()
    if not ret:
        break

    results = model(frame).pandas().xyxy
    scores = results[0]['confidence'].to_numpy(dtype=np.float32)
    detections = results[0][['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_numpy(dtype=np.float32)
    classes = results[0]['class'].to_numpy()

    if len(detections):
        # centernet will do detection on all the COCO classes. "person" is class number 0
        idxs = np.where(classes == 0)[0]
        detections = detections[idxs]
        scores = scores[idxs]
        classes = classes[idxs]
    else:
        boxes = np.empty((0, 5))

    res = sort.update(detections)

    boxes_track = res[:,:-1].astype(int)
    boxes_ids = res[:,-1].astype(int)

    for bbox, id, score in zip(boxes_track, boxes_ids, scores):
        text = "{} ID:{} Score:{:.2f}".format(DETECT, id, score)
        cv2.putText(frame, text, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)  # Annotate the frame to be sent back to the main driver
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

    i += 1
    cv2.imwrite("./track/frame{}.png".format(i), frame)