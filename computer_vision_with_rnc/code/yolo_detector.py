from keras.models import load_model
import cv2
import numpy as np
from utils import sigmoid


class YoloDetector:
    """
    Represents an object detector for robot soccer based on the YOLO algorithm.
    """
    def __init__(self, model_name, anchor_box_ball=(5, 5), anchor_box_post=(2, 5)):
        """
        Constructs an object detector for robot soccer based on the YOLO algorithm.

        :param model_name: name of the neural network model which will be loaded.
        :type model_name: str.
        :param anchor_box_ball: dimensions of the anchor box used for the ball.
        :type anchor_box_ball: bidimensional tuple.
        :param anchor_box_post: dimensions of the anchor box used for the goal post.
        :type anchor_box_post: bidimensional tuple.
        """
        self.network = load_model(model_name + '.hdf5')
        self.network.summary()  # prints the neural network summary
        self.anchor_box_ball = anchor_box_ball
        self.anchor_box_post = anchor_box_post

    def detect(self, image):
        """
        Detects robot soccer's objects given the robot's camera image.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        image = self.preprocess_image(image)
        output = self.network.predict(image)
        # print(output)
        return self.process_yolo_output(output)
        # Todo: implement object detection logic
        # ball_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: remove this line
        # post1_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: remove this line
        # post2_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: remove this line
        # return ball_detection, post1_detection, post2_detection

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        image = cv2.resize(image, (120, 160), interpolation=cv2.INTER_AREA)
        image = np.array(image) / 255.0
        image = np.reshape(image, (1, 120, 160, 3))
        return image

    def process_yolo_output(self, output):
        """
        Processes the neural network's output to yield the detections.

        :param output: neural network's output.
        :type output: NumPy 4-dimensional array with dimensions (1, 15, 20, 10).
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        coord_scale = 4 * 8  # coordinate scale used for computing the x and y coordinates of the BB's center
        bb_scale = 640  # bounding box scale used for computing width and height
        output = np.reshape(output, (15, 20, 10))  # reshaping to remove the first dimension

        max_ball_prob = 0
        max_ball_row_idx = 0
        max_ball_col_idx = 0

        max_cross_first_prob = 0
        max_cross_first_row_idx = 0
        max_cross_first_col_idx = 0
        max_cross_sec_prob = 0
        max_cross_sec_row_idx = 0
        max_cross_sec_col_idx = 0

        for row_idx, row in enumerate(output):
            for elm_idx, elm in enumerate(row):

                # treat ball case:
                ball_prob = sigmoid(elm[0])
                if ball_prob > max_ball_prob:
                    max_ball_prob = ball_prob
                    max_ball_row_idx = row_idx
                    max_ball_col_idx = elm_idx

                # treat crossbar case:
                # ball_prob = sigmoid(elm[5])
                # if ball_prob > max_ball_prob:
                #     max_ball_prob = ball_prob
                #     max_ball_row_idx = row_idx
                #     max_ball_col_idx = elm_idx

        # print(output)

        # Todo: implement YOLO logic
        ball_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: change this line
        post1_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: change this line
        post2_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: change this line

        x_ball = max_ball_col_idx*coord_scale
        y_ball = max_ball_row_idx*coord_scale
        ball_detection = (max_ball_prob, x_ball, y_ball, coord_scale, coord_scale)

        return ball_detection, post1_detection, post2_detection
