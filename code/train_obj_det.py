#%% Import libraries
# import tensorflow as tf 
import torch
from ultralytics import YOLO

# Ensure pytorch is using GPU/CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# # For every gpu utilized, limit memory growth
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#%% Train Object Detection Model (yolov8)

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# Use the model
results = model.train(data="./Main Program/config.yaml", epochs=30, device='cpu')  # train the model

#%% Validate Object Detection Model
# Load a model
model = YOLO('./runs/detect/train9/weights/best.pt')  # load a custom model
# Validate the model
metrics = model.val(device='cpu')  # evaluate model performance on the validation set

#%% Predict on Object Detection Model on Test Images

# Load a pretrained YOLOv8n model
model = YOLO('./runs/detect/train9/weights/best.pt')  # pretrained YOLOv8n model

# Define path to the image file
source = './object detection data/images/test/Donor6035.jpg'

# Run inference on the source
results = model(source)  # list of Results objects