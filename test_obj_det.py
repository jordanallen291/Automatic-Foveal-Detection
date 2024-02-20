from ultralytics import YOLO
import glob
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Ensure pytorch is using GPU/CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


#%% Function to locate fovea in images
def locate_fovea(image_path, model):
    # Run inference on the source
    results = model.predict(image_path)  # list of Results objects
    
    # Show the results
    for result in results:
        if result.boxes.cpu().numpy().xyxy.shape[0] != 0:  # Check if the fovea was found
            im_array = result.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
            # Display the image and draw a vertical line where center_line is found
            plt.imshow(im)
            # plt.axvline(x=center_line, color='r', linestyle='--')
            plt.show()
        else:
            print('No fovea found')
    return 

# Load a pretrained YOLOv8n model
model = YOLO('./runs/detect/train9/weights/best.pt')  # pretrained YOLOv8n model

# Define path to the 'test' images directory
test_images_dir = './object detection data/images/test/'

# List all jpg images in the 'test' directory
image_files = glob.glob(test_images_dir + '*.jpg')

# Process each image in the directory
for image_path in image_files:
    print(f"Processing {image_path}")
    center_line = locate_fovea(image_path, model)
