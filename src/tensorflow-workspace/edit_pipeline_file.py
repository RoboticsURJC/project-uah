import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

PATH_TO_PIPELINE_CONFIG = "/home/jamarma/research/projects/project-uah/src/tensorflow-works/workspace/models/ssdmobilenetv2-fpnlite-640/v1/pipeline.config"
PATH_TO_LABEL_MAP = "/home/jamarma/research/projects/project-uah/src/tensorflow-works/workspace/data/label_map.pbtxt"
PATH_TO_TRAIN_RECORD = "/home/jamarma/research/projects/project-uah/src/tensorflow-works/workspace/data/train.tfrecord"
PATH_TO_TEST_RECORD = "/home/jamarma/research/projects/project-uah/src/tensorflow-works/workspace/data/test.tfrecord"
PATH_TO_PRETRAINED_CHECKPOINT = "/home/jamarma/research/projects/project-uah/src/tensorflow-works/workspace/pre_trained_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"

NUM_CLASSES = 1
BATCH_SIZE = 16
DETECTION_TYPE = "detection" # Object detection
NUM_STEPS = 5000

# proto_str variable for modify variables of pipeline file
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(PATH_TO_PIPELINE_CONFIG, "r") as f:
  proto_str = f.read()
  text_format.Merge(proto_str, pipeline_config)

# Editing pipeline file
pipeline_config.model.ssd.num_classes = NUM_CLASSES
pipeline_config.train_config.batch_size = BATCH_SIZE
pipeline_config.train_config.fine_tune_checkpoint = PATH_TO_PRETRAINED_CHECKPOINT
pipeline_config.train_config.num_steps = NUM_STEPS
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = NUM_STEPS
pipeline_config.train_config.fine_tune_checkpoint_type = DETECTION_TYPE
pipeline_config.train_input_reader.label_map_path = PATH_TO_LABEL_MAP
pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = PATH_TO_TRAIN_RECORD
pipeline_config.eval_input_reader[0].label_map_path = PATH_TO_LABEL_MAP
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = PATH_TO_TEST_RECORD

# Save pipeline file
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(PATH_TO_PIPELINE_CONFIG, "wb") as f:
  f.write(config_text)