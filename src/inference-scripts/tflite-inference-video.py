# This code is based in: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_video.py

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from utils.fps import FPS
import time

MODEL_PATH = "../tensorflow-workspace/tflite/models/ssdmobilenetv2-fpnlite-320/model-v4.tflite"
TEST_VIDEO_PATH = "../tensorflow-workspace/data/data_to_test/videos/test.mp4"
LABELMAP_PATH = "../tensorflow-workspace/data/label-map.txt"

MIN_CONF_THRESHOLD = 0.5

# Load the label map
with open(LABELMAP_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
boxes_idx, classes_idx, scores_idx = 1, 3, 0

# Open video file
video = cv2.VideoCapture(TEST_VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = FPS().start()

while(video.isOpened()):
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Bounding box coordinates of detected objects
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    # Class index of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    # Confidence of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Loop over all detections and draw detection box 
    # if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions,
            # need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # Show the FPS
    fps.update()
    fps_text = 'FPS = {:.1f}'.format(fps.fps())
    cv2.putText(frame, fps_text, (24, 20), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 0, 255), 1)

    cv2.imshow('Test image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

fps.stop()
print("Tiempo aproximado por frame {}".format(fps.fps()))
video.release()
cv2.destroyAllWindows()