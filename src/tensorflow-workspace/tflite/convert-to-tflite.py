import tensorflow as tf
import os

MODEL = "ssdmobilenetv2-fpnlite-320"
VERSION = "v4"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(
    "exported-models/"+MODEL+"/"+VERSION+"/saved_model")
tflite_model = converter.convert()

# Create model folder
model_folder_path = "models/"+MODEL
if not os.path.exists(model_folder_path):
    os.mkdir(model_folder_path)

# Save the model
with open(model_folder_path+"/model-"+VERSION+".tflite", 'wb') as f:
    f.write(tflite_model)

