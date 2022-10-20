import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

import tensorflow as tf
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import imutils
import time
import dlib
from imutils.video import VideoStream
from imutils.video import FPS

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_MODEL_DIR = "../tensorflow-works/workspace/exported_models/ssdmobilenetv2/v1"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

# Load labelmap
label_map_pbtxt_fname = "../tensorflow-works/workspace/data/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map_pbtxt_fname)

# Path video
PATH_VIDEO = "../tensorflow-works/workspace/data/data_to_test/videos/test.mp4"

vs = cv2.VideoCapture(PATH_VIDEO)

# To measure FPS
fps = FPS().start()

while True:
    # Read frames
    ret, frame = vs.read()

    if frame is None:
        break

    # Converting to array
    image_np = np.array(frame)

    # Converting to tensor and adding new dimension
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Make object detection
    detections = detect_fn(input_tensor)

    # Analizing num of detections
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0,:num_detections].numpy() for key, value in detections.items()}

    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Cleaning detections with treshold
    #detection_scores = np.array(detections["detection_scores"][0])
    #detection_clean = [x for x in detection_scores if x >= TRESHOLD]

    # Copy image to draw bounding box
    image_np_with_detections = image_np.copy()

    # Object detection API to visualize bounding box
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        max_boxes_to_draw=200,
        min_score_thresh=0.50,
        use_normalized_coordinates = True
    )

    cv2.imshow("Test", image_np_with_detections)
    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

fps.stop()

print("Tiempo completo {}".format(fps.elapsed()))
print("Tiempo aproximado por frame {}".format(fps.fps()))

cv2.destroyAllWindows()
vs.release()