import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from utils import utils

# Paths to files
MODEL_PATH = "../tensorflow-workspace/tflite/models-with-metadata/ssdmobilenetv2-fpnlite-320/model-v4.tflite"
TEST_IMAGE_PATH = "../tensorflow-workspace/data/data_to_test/images/0001.png"
LABELMAP_PATH = "../tensorflow-workspace/data/label-map.txt"

# Parameter values of input image in MobileNet
INPUT_MEAN = 127.5
INPUT_STD = 127.5

# Minimun value of confidence threshold
MIN_CONF_THRESHOLD = 0.5

def preprocess_image(image_path, size):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, size)
    resized_image = np.expand_dims(resized_image, axis=0)
    normalized_image = (np.float32(resized_image) - INPUT_MEAN) / INPUT_STD
    return image, normalized_image

def detect(interpreter, image, min_threshold):
    tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(tensor_index, image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    count = int(interpreter.get_tensor(output_details[2]['index'])[0])
    classes = interpreter.get_tensor(output_details[3]['index'])[0]

    results = []
    for i in range(count):
        if scores[i] >= min_threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

# Load labels from label-map
with open(LABELMAP_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input shape required by the model
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# Preprocess image
original_image, processed_image = preprocess_image(TEST_IMAGE_PATH, (input_height, input_width))

# Run object detection
results = detect(interpreter, processed_image, MIN_CONF_THRESHOLD)

# Draw bounding boxes
original_image = utils.visualize(original_image, results, labels)

cv2.imshow('Test image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()