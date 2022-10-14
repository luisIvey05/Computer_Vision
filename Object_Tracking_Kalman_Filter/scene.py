import numpy as np
from filterpy.kalman import KalmanFilter
from vehicle import Vehicle

np.random.seed(0)


def linear_assignment(cost_matrix):
    """
    linear_assignment: Using the cost_matrix, linear_assignment will return a nx2 numpy array that assigns all the new
    detections that were made to the current vehicles being tracked.
    :param cost_matrix: intersection over union matrix between new and old vehicles being tracked.
    :return nx2 numpy array: assignment between the old and new vehicles that match.
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou(bb_new, bb_old):
    """
    iou: calculates the intersection over union between two lists that for each index the values are bbox points.
    :param bb_new: new detections list.
    :param bb_old: previously tracked vehicles.
    """
    bb_old = np.expand_dims(bb_old, 0)
    bb_new = np.expand_dims(bb_new, 1)

    xx1 = np.maximum(bb_new[..., 0], bb_old[..., 0])
    yy1 = np.maximum(bb_new[..., 1], bb_old[..., 1])
    xx2 = np.minimum(bb_new[..., 2], bb_old[..., 2])
    yy2 = np.minimum(bb_new[..., 3], bb_old[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_new[..., 2] - bb_new[..., 0]) * (bb_new[..., 3] - bb_new[..., 1])
              + (bb_old[..., 2] - bb_old[..., 0]) * (bb_old[..., 3] - bb_old[..., 1]) - wh)
    return o


def bbox_to_tracker(detections, trackers, iou_threshold=0.3):
    """
    bbox_to_tracker: Trys to map new detections to current vehicles being tracked.
    :param detections: a list of new bbox locations nx4.
    :param trackers: a list of previous vehicles being tracked nx4.
    :param iou_threshold: the location between the new locations and predicted locations must be at or above a
    specified threshold.
    :return matches, unmatched_vehicles: returns two lists. matches is a nx2 where index[0]= the index of the new
    detection and index[1]= the index of tracker list where a vehicle that was previously tracked.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections))

    iou_matrix = iou(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections)


class VehicleTracker(object):
    """
    VehicleTracker: a class that keeps track of vehicles in a scene.
    """
    def __init__(self, age_threshold, hits_threshold, iou_threshold, previous_centroid, box_size):
        self.tracked_vehicles = []
        self.tracked_stopped_vehicles = {}
        self.frame_num = 0
        self.age_threshold = age_threshold
        self.hits_threshold = hits_threshold
        self.iou_threshold = iou_threshold
        self.previous_centroid = previous_centroid
        self.box_size = box_size

    def update(self, frame_no, new_detections=np.empty((0, 6))):
        """
        :param new_detections: a list of new detections made by YOLOv5.
        :return [xmin, ymin, xmax, ymax, confidence, isStationary]: a list of vehicle attributes.
        """
        current_vehicles_being_tracked = np.zeros((len(self.tracked_vehicles), 6))
        vehicles_no_longer_in_scene = []

        for idx, vehicle in enumerate(current_vehicles_being_tracked):
            position = self.tracked_vehicles[idx].predict()
            vehicle[:] = [position[0], position[1], position[2], position[3], 0, 0]
            if np.any(np.isnan(position)):
                vehicles_no_longer_in_scene.append(idx)

        current_vehicles_being_tracked = np.ma.compress_rows(np.ma.masked_invalid(current_vehicles_being_tracked))

        for idx in reversed(vehicles_no_longer_in_scene):
            self.tracked_vehicles.pop(idx)

        matched_vehicles, unmatched_vehicles = bbox_to_tracker(
            new_detections, current_vehicles_being_tracked, self.iou_threshold)

        for vehicle_idx in matched_vehicles:
            self.tracked_vehicles[vehicle_idx[1]].update(new_detections[vehicle_idx[0], :])

        for new_vehicle_idx in unmatched_vehicles:
            new_vehicle = Vehicle(new_detections[new_vehicle_idx, :], self.previous_centroid, self.box_size)
            self.tracked_vehicles.append(new_vehicle)

        self.frame_num += 1
        i = len(self.tracked_vehicles)
        output_vehicles = []

        for vehicle in reversed(self.tracked_vehicles):
            d = vehicle.get_state()
            isStationary = vehicle.check_stationary(frame_no)
            if isStationary and (vehicle.id + 1) not in self.tracked_stopped_vehicles:
                self.tracked_stopped_vehicles[(vehicle.id + 1)] = vehicle
            i -= 1

            if (vehicle.time_since_update < 1) and (vehicle.hit_streak >= self.hits_threshold or
                                                    self.frame_num <= self.hits_threshold):
                output_vehicles.append(np.concatenate((d, [vehicle.id + 1], [isStationary])).reshape(1, -1))

            if vehicle.time_since_update > self.age_threshold:
                self.tracked_vehicles.pop(i)

        if len(output_vehicles) > 0:
            return np.concatenate(output_vehicles)

        return np.empty((0, 5))

    def get_vehicle_information(self, id):
        start = self.tracked_stopped_vehicles[id].get_start_frame()
        end   = self.tracked_stopped_vehicles[id].get_stop_frame()
        location = self.tracked_stopped_vehicles[id].get_stopped_location().astype(int)

        return start, end, location