import numpy as np
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((4,))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.previous = 2
        self.centerX = [0] * self.previous
        self.centerY = [0] * self.previous
        self.rolling_counter = 0
        self.center = [0, 0]
        self.isStationary = False
        self.box_size = 20

    def update(self, bbox):
        """
    Updates the state vector with observed bbox.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        x, y = self.calculate_center(bbox)
        self.store_center(x, y)
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return convert_x_to_bbox(self.kf.x)

    def check_stationary(self):
        if self.isStationary:
            centerX, centerY = self.center[0], self.center[1]
            curr = self.get_state()
            currX, currY = self.calculate_center(curr)

            if currX < centerX - self.box_size or currX > centerX + self.box_size or \
                    currY < centerY - self.box_size or currY > centerY + self.box_size:
                self.isStationary = False
                return self.isStationary
            else:
                return self.isStationary

        elif self.centerX.count(self.centerX[0]) == len(self.centerX) and \
                self.centerY.count(self.centerY[0]) == len(self.centerY) and self.hit_streak >= len(self.centerX) - 1:
            self.isStationary = True
            self.center[:] = [self.centerX[0], self.centerY[0]]
            return self.isStationary

        else:
            return self.isStationary

    def calculate_center(self, bbox):
        x = round((abs(bbox[2] - bbox[0])) / 2)
        y = round((abs(bbox[3] - bbox[1])) / 2)
        return x, y

    def store_center(self, x, y):
        self.centerY[self.rolling_counter] = y
        self.centerX[self.rolling_counter] = x
        self.rolling_counter = (self.rolling_counter + 1) % self.previous


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

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
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class VehicleTracker(object):
    def __init__(self, max_age, min_hits, iou_threshold):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.vehicles = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 6))):
        self.frame_count += 1
        vecs = np.zeros((len(self.vehicles), 6))
        to_del = []
        ret = []
        for idx, vec in enumerate(vecs):
            pos = self.vehicles[idx].predict()
            vec[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(idx)
        vecs = np.ma.compress_rows(np.ma.masked_invalid(vecs))
        for idx in reversed(to_del):
            self.vehicles.pop(idx)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, vecs, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.vehicles[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.vehicles.append(trk)

        i = len(self.vehicles)
        for trk in reversed(self.vehicles):
            d = trk.get_state()
            isStationary = trk.check_stationary()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [isStationary])).reshape(1,
                                                                                     -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.vehicles.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))
