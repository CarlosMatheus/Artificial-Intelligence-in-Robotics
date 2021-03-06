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
        return self.process_yolo_output(output)

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
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

        def get_crossbar_params(row_idx, elm_idx, elm):
            x_cross = (elm_idx + sigmoid(elm[6])) * coord_scale
            y_cross = (row_idx + sigmoid(elm[7])) * coord_scale
            w_cross = bb_scale * 2 * np.exp(elm[8])
            h_cross = bb_scale * 5 * np.exp(elm[9])
            return x_cross, y_cross, w_cross, h_cross

        ball_detection = (0.0, 0.0, 0.0, 0.0, 0.0)
        post1_detection = (0.0, 0.0, 0.0, 0.0, 0.0)
        post2_detection = (0.0, 0.0, 0.0, 0.0, 0.0)

        for row_idx, row in enumerate(output):
            for elm_idx, elm in enumerate(row):

                # treat ball case:
                ball_prob = sigmoid(elm[0])
                if ball_prob > ball_detection[0]:
                    x_ball = (elm_idx + sigmoid(elm[1]) )* coord_scale
                    y_ball = (row_idx + sigmoid(elm[2]) )* coord_scale
                    w_ball = bb_scale * 5 * np.exp(elm[3])
                    h_ball = bb_scale * 5 * np.exp(elm[4])
                    ball_detection = (ball_prob, x_ball, y_ball, w_ball, h_ball)

                # treat crossbar case:
                cross_prob = sigmoid(elm[5])
                if cross_prob > post1_detection[0]:
                    x_cross, y_cross, w_cross, h_cross = get_crossbar_params(row_idx, elm_idx, elm)
                    post1_detection = (cross_prob, x_cross, y_cross, w_cross, h_cross)
                elif cross_prob > post2_detection[0]:
                    x_cross, y_cross, w_cross, h_cross = get_crossbar_params(row_idx, elm_idx, elm)
                    post2_detection = (cross_prob, x_cross, y_cross, w_cross, h_cross)

        return ball_detection, post1_detection, post2_detection
