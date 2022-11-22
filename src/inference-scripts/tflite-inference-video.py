import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from utils.fps import FPS
from utils import utils
import time

# Paths to files
MODEL_PATH = "../tensorflow-workspace/tflite/models/ssdmobilenetv2-fpnlite-320/model-v4.tflite"
TEST_VIDEO_PATH = "../tensorflow-workspace/data/data_to_test/videos/test.mp4"
LABELMAP_PATH = "../tensorflow-workspace/data/label-map.txt"

# Parameter values of input image in MobileNet
INPUT_MEAN = 127.5
INPUT_STD = 127.5

# Minimun value of confidence threshold
MIN_CONF_THRESHOLD = 0.5

def preprocess_frame(frame, size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, size)
    resized_frame = np.expand_dims(resized_frame, axis=0)
    normalized_frame = (np.float32(resized_frame) - INPUT_MEAN) / INPUT_STD
    return normalized_frame

def detect(interpreter, frame, min_threshold):
    tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(tensor_index, frame)
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

# Load the label map
with open(LABELMAP_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input shape required by the model
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# Open video file
video = cv2.VideoCapture(TEST_VIDEO_PATH)
video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = FPS().start()

while(video.isOpened()):
    ret, frame = video.read()
    if not ret:
      break
    # Preprocess frame
    processed_frame = preprocess_frame(frame, (input_height, input_width))

    # Run object detection
    results = detect(interpreter, processed_frame, MIN_CONF_THRESHOLD)

    # Draw bounding boxes
    frame = utils.visualize(frame, results, labels)

    # Show the FPS
    fps.update()
    fps_text = 'FPS = {:.1f}'.format(fps.fps())
    cv2.putText(frame, fps_text, (24, 20), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 0, 255), 1)

    cv2.imshow('Test video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

fps.stop()
print("Tiempo aproximado por frame {}".format(fps.fps()))

video.release()
cv2.destroyAllWindows()