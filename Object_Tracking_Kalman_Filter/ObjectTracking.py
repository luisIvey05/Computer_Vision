import sys
import os
from google.colab import files
import centernet

sys.path.append("/content/ObjectTracking")
from sort import Sort
from lib import VisTrack, show_video, create_video

import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import shutil


video_file = "./video.mp4"
# Default: num_classes=80
obj = centernet.ObjectDetection(num_classes=80)

# num_classes=80 and weights_path=None: Pre-trained COCO model will be loaded.
obj.load_weights(weights_path=None)

vidcap = cv2.VideoCapture(video_file)
fps = vidcap.get(cv2.CAP_PROP_FPS)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

folder_out = "Track"
if os.path.exists(folder_out):
    shutil.rmtree(folder_out)
os.makedirs(folder_out)

draw_imgs = []

sort = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

pbar = tqdm(total=length)
i = 0
while True:
    ret, frame = vidcap.read()
    if not ret:
        break

    boxes, classes, scores = obj.predict(frame)
    detections_in_frame = len(boxes)
    if detections_in_frame:
        # centernet will do detection on all the COCO classes. "person" is class number 0
        idxs = np.where(classes == 0)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]
        classes = classes[idxs]
    else:
        boxes = np.empty((0, 5))

    dets = np.hstack((boxes, scores[:,np.newaxis]))
    res = sort.update(dets)

    boxes_track = res[:,:-1]
    boces_ids = res[:,-1].astype(int)

    p_frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if detections_in_frame:
        p_frame = vt.draw_bounding_boxes(p_frame, boxes_track, boces_ids, scores)
    p_frame.save(os.path.join(folder_out, f"{i:03d}.png"))

    i+=1
    pbar.update(1)