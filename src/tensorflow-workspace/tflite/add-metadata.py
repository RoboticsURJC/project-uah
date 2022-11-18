from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata
import os

MODEL = "ssdmobilenetv2-fpnlite-320"
VERSION = "v4"
LABEL_FILE_PATH = "../data/label-map.txt"

model_path = "models/"+MODEL+"/model-"+VERSION+".tflite"
save_to_path = "models-with-metadata/"+MODEL+"/model-"+VERSION+".tflite"
if not os.path.exists("models-with-metadata/"+MODEL):
    os.mkdir("models-with-metadata/"+MODEL)

ObjectDetectorWriter = object_detector.MetadataWriter

writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(model_path), input_norm_mean=[127.5], 
    input_norm_std=[127.5], label_file_paths=[LABEL_FILE_PATH])
writer_utils.save_file(writer.populate(), save_to_path)