import numpy as np
from filterpy.kalman import KalmanFilter


def calculate_center(bbox):
    """
    calculate_center: Using the four points passed in the list (bbox), calculate_center returns the center point of
    the bounding box.
    :param bbox: a list of bounding points. xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3]
    :return (x, y): a tuple that indicates the center position of a bounding box
    """
    x = round((abs(bbox[2] - bbox[0])) / 2)
    y = round((abs(bbox[3] - bbox[1])) / 2)
    return x, y


def convert_bbox_to_z(bbox):
    """
    convert_bbox_to_z: Using bbox=[xmin, ymin, xmax, ymax], convert_bbox_to_z outputs a converted representation of the
    bbox. x, y = center coordinates of the bbox. a = the area of the bbox. r = aspect ratio.
    :return z=numpy array 4x1: [x, y = center coordinates of the bbox. a = the area of the bbox. r = aspect ratio.]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    a = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, a, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    convert_x_to_bbox: Using the z representation of a bbox, convert_x_to_bbox converts z coordinates to [xmin, ymin,
    xmax, ymax] representation.
    :param x: z representation of a bbox.
    :param score: confidence level that it is a vehicle.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((4,))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class Vehicle(object):
    """
    Vehicle: Keeps track of a vehicle based on their bbox and holds different attributes of the vehicle.
    """
    count = 0

    def __init__(self, bbox, previous_centroid, box_size):
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = Vehicle.count
        Vehicle.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.previous_centroid = previous_centroid
        self.centerX = [0] * self.previous_centroid
        self.centerY = [0] * self.previous_centroid
        self.rolling_counter = 0
        self.center = [0, 0]
        self.isStationary = False
        self.box_size = box_size
        self.stop_frame = 0
        self.start_frame = 0
        self.first_stopped = False
        self.stopped_location = []

    def check_stationary(self, frame_num):
        """
        check_stationary: checks to see if the vehicle is stationary by checking if it had a specified number of same
        bbox locations or checks to see if its within a certain range based on a previous stationary point.
        :return self.isStationary: a bool attribute that says if the vehicle is parked or not"""
        if self.isStationary:
            centerX, centerY = self.center[0], self.center[1]
            curr = self.get_state()
            currX, currY = calculate_center(curr)

            if currX < centerX - self.box_size or currX > centerX + self.box_size or \
                    currY < centerY - self.box_size or currY > centerY + self.box_size:
                self.isStationary = False
                if self.first_stopped:
                    self.stop_frame = frame_num
                    self.first_stopped = False
                return self.isStationary
            else:
                return self.isStationary

        elif self.centerX.count(self.centerX[0]) == len(self.centerX) and \
                self.centerY.count(self.centerY[0]) == len(self.centerY) and self.hit_streak >= len(self.centerX) - 1:
            self.isStationary = True

            if not self.first_stopped:
                self.stopped_location = self.get_state() + 10
                self.start_frame = frame_num
                self.first_stopped = True
            self.center[:] = [self.centerX[0], self.centerY[0]]
            return self.isStationary

        else:
            return self.isStationary

    def store_center(self, x, y):
        """
        store_center: round-robin format of storing a center location.
        :param x: x-axis center location
        :param y: y-axis center location
        """
        self.centerY[self.rolling_counter] = y
        self.centerX[self.rolling_counter] = x
        self.rolling_counter = (self.rolling_counter + 1) % self.previous_centroid

    def update(self, bbox):
        """
        update: Using the passed in bbox, update will update the new location of the vehicle in z representation.
        :param bbox: bounding box of the vehicles new location.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        x, y = calculate_center(bbox)
        self.store_center(x, y)
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        predict: Advances the state vector and returns the predicted bounding box estimate.
        :return new bbox 4x1: new predicted bounding box estimation.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """
        get_state: returns the current location of the vehicle based on its convert z to bbox representation.
        :return current bbox 4x1: current location of the vehicle.
        """
        return convert_x_to_bbox(self.kf.x)

    def set_start_frame(self, frame_num):
        self.start_frame = frame_num

    def get_start_frame(self):
        return self.start_frame

    def set_stop_frame(self, frame_num):
        self.stop_frame = frame_num

    def get_stop_frame(self):
        return self.stop_frame

    def get_stopped_location(self):
        return self.stopped_location
