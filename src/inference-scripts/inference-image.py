import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_MODEL_DIR = "../tensorflow-workspace/exported_models/ssdmobilenetv2/v1"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

# Load labelmap
label_map_pbtxt_fname = "../tensorflow-workspace/data/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map_pbtxt_fname)

# Test image path
image_path = "../tensorflow-workspace/data/data_to_test/images/0001.png"

# Converting to array
image_np = np.array(Image.open(image_path))

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

# Copy image to draw bounding box
image_np_with_detections = image_np.copy()

# Object detection API to visualize bounding box
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    max_boxes_to_draw=10,
    min_score_thresh=0.50,
    use_normalized_coordinates = True
)

cv2.imshow("Test", image_np_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()