import os
#from util import bounding
from sort import VehicleTracker
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import torch
import cv2
import shutil


VIDEO_FILE = "./video.mp4"
DETECT = "car"
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

tracker = VehicleTracker(max_age=1, min_hits=3, iou_threshold=0.3)

i = 0
rolling_counter = 0
pointx = [0] * 12
pointy = pointx
hasStopped = 0
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
        idxs = np.where(classes == 2)[0]
        detections = detections[idxs]
        scores = scores[idxs]
        classes = classes[idxs]
    else:
        boxes = np.empty((0, 5))

    res = tracker.update(detections)

    boxes_track = res[:, :-2].astype(int)
    boxes_ids = res[:, -2].astype(int)
    stopped_vehicles = res[:, -1].astype(bool)

    for bbox, id, score, stopped in zip(boxes_track, boxes_ids, scores, stopped_vehicles):
        text = "{} ID:{}".format(DETECT, id)
        if stopped:
            cv2.putText(frame, text, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255),
                        1)  # Annotate the frame to be sent back to the main driver
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
        else:
            cv2.putText(frame, text, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0),
                        1)  # Annotate the frame to be sent back to the main driver
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
    #     if id == 1:
    #         x = round((abs(bbox[2] - bbox[0])) / 2)
    #         y = round((abs(bbox[3] - bbox[1])) / 2)
    #         pointy[rolling_counter] = y
    #         pointx[rolling_counter] = x
    #         rolling_counter = (rolling_counter + 1) % 12
    #         if pointy.count(pointy[0]) == len(pointy) and pointx.count(pointx[0]) == len(pointx) and not hasStopped:
    #             hasStopped = True


    cv2.imshow("Frame", frame)  # Display anotated frame from the suspicious class
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


    i += 1
    # cv2.imwrite("./track/frame{}.png".format(i), frame)