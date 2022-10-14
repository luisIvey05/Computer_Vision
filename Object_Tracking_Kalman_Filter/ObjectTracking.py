import os
# from util import bounding
from scene import VehicleTracker
from action_recognition import cropped, action
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import torch
import cv2
import shutil

VIDEO_FILE = "./video.mp4"
ACTION_FILE = ""
DETECT = [2, 3, 5, 7]
DETECTION_THRESHOLD = 0.5
MAX_AGE = 20
MIN_HITS = 3
IOU_THRESHOLD = 0.1
PREVIOUS_CENTROID = 3
BOX_SIZE = 40
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

scene = VehicleTracker(MAX_AGE, MIN_HITS, IOU_THRESHOLD, PREVIOUS_CENTROID, BOX_SIZE)

frame_no = 1
rolling_counter = 0
pointx = [0] * 12
pointy = pointx
hasStopped = 0
previous_stopped_vehicles = []

while True:
    ret, frame = vidcap.read()
    if not ret:
        break

    results = model(frame).pandas().xyxy
    # print(results)
    scores = results[0]['confidence'].to_numpy(dtype=np.float32)
    detections = results[0][['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_numpy(dtype=np.float32)
    classes = results[0]['class'].to_numpy()

    if len(detections):
        # centernet will do detection on all the COCO classes. "person" is class number
        idxs = np.where(scores > DETECTION_THRESHOLD)
        # print(idxs)
        classes = classes[idxs]
        detections = detections[idxs]
        scores = scores[idxs]
        # print(scores)
        # print(classes)
        idxs = np.in1d(classes, DETECT)
        # print(idxs)
        idxs = np.where(idxs == True)
        # print(idxs)
        detections = detections[idxs]
        scores = scores[idxs]
        classes = classes[idxs]
    else:
        boxes = np.empty((0, 5))

    res = scene.update(frame_no, detections)

    boxes_track = res[:, :-2].astype(int)
    boxes_ids = res[:, -2].astype(int)
    stopped_vehicles = res[:, -1].astype(bool)

    for bbox, id, score, stopped in zip(boxes_track, boxes_ids, scores, stopped_vehicles):
        text = "Stopped Vehicle ID:{}".format(id)
        if stopped:
            cv2.putText(frame, text, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255),
                        1)  # Annotate the frame to be sent back to the main driver
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

            if id not in previous_stopped_vehicles:
                previous_stopped_vehicles.append(id)

        else:
            if id in previous_stopped_vehicles:
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

    frame_no += 1

vidcap.release()
cv2.destroyAllWindows()

for vehicle in previous_stopped_vehicles:
    start, end, bbox = scene.get_vehicle_information(vehicle)
    cropped(VIDEO_FILE, start, end, bbox)
    action(ACTION_FILE)

# start = scene.return_suspicious_vehicle[0].get_stationary_frame()  # Grab the frame no. where we first detected the
# end = scene.return_suspicious_vehicle[0].get_nonstationary_frame() # stationary vehicle and the frame no. when the
# location = scene.return_suspicious_vehicle[0].get_bounding()       # vehicle moved
# print(location)
# cropped(filename, start, end, location) # Crop the video to the passed in ROI area
# action("/projectnb/ec720prj/SVDetect/EC520_SVD/output.avi") # The new cropped video is passed to the action rec.
#                                                             # pipeline for action recognition.
# cv2.imwrite("./track/frame{}.png".format(i), frame)
