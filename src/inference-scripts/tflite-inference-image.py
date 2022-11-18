from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

MODEL_PATH = "../tensorflow-workspace/tflite/models-with-metadata/ssdmobilenetv2-fpnlite-320/model-v4.tflite"
TEST_IMAGE_PATH = "../tensorflow-workspace/data/data_to_test/images/0001.png"

# base_options = core.BaseOptions(file_name=MODEL_PATH)
# detection_options = processor.DetectionOptions(max_results=2)
# options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
# detector = vision.ObjectDetector.create_from_options(options)

detector = vision.ObjectDetector.create_from_file(MODEL_PATH)

image = vision.TensorImage.create_from_file(TEST_IMAGE_PATH)
detection_result = detector.detect(image)
print(detection_result.detections)