#%% Import libraries
import pyautogui as pag
import time
import random
import matplotlib.pyplot as plt
import winsound
import os
import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PIL import Image

# Ensure pytorch is using GPU/CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
# For every gpu utilized, limit memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#%% Acquire images

def acquire_images():
    # Wait to switch screens after starting program
    time.sleep(2)
    
    freq=1000
    dur=100
    
    # Define the coordinates of the region to monitor
    # These are dummy coordinates
    x1, y1, x2, y2 = 1232, 1587, 2135, 1991
    
    # Define the color you're looking for in RGB format
    target_color = (212, 208, 200)
    
    # Set state of acquiring images
    acquire = False
    while acquire == False:
        # Capture a screenshot of the region
        screenshot = pag.screenshot(region=(x1, y1, x2-x1, y2-y1))
        plt.imshow(screenshot)
        # Get the color of the pixel at the center of the region
        pixel_color = screenshot.getpixel((screenshot.width // 2, screenshot.height // 2))
        print(pixel_color)
    
        # If the color of the pixel does not match the target, move to acquire phase
        if pixel_color != target_color:
            print("Acquiring Images")
            winsound.Beep(freq,dur)
            acquire = True
        # If the color does match the target, wait for acquire phase
        else:
            print("Waiting to Acquire")
            # Wait for a short period before checking again
            time.sleep(0.5)
            
    while acquire == True:
        # Capture a screenshot of the region
        screenshot = pag.screenshot(region=(x1, y1, x2-x1, y2-y1))
        plt.imshow(screenshot)
        # Get the color of the pixel at the center of the region
        pixel_color = screenshot.getpixel((screenshot.width // 2, screenshot.height // 2))
        print(pixel_color)
    
        # If the color of the pixel matches the target color, acquire has ended
        if pixel_color == target_color:
            print("Acquisition Ended")
            winsound.Beep(freq,dur)
            acquire = False
        # If the color does not match, Images are still being acquired
        else:
            print("Acquiring Images")
            # Wait for a short period before checking again
            time.sleep(0.5)
    print("Examining Images")
    
acquire_images()
#%% Save images
def save_images():
    # Save image button
    save_images_btn = (690, 108)
    # Change to desktop directory
    desktop_btn = (87, 216)
    # New Folder
    folder_btn = (150, 125)
    # Save button location
    save_btn = (591, 469)
    
    # name of image
    image_name = (f"{random.randint(1000, 100000)}")
    
    # Execute movement
    pag.leftClick(save_images_btn[0],save_images_btn[1])
    time.sleep(0.3)
    pag.leftClick(desktop_btn[0], desktop_btn[1])
    time.sleep(0.3)
    pag.leftClick(folder_btn[0], folder_btn[1])
    time.sleep(0.3)
    # Write figure name
    pag.write(image_name, .1)
    pag.press("enter")
    time.sleep(0.3)
    pag.press("enter")
    time.sleep(0.3)
    pag.leftClick(save_btn[0], save_btn[1])
    time.sleep(0.3)

save_images()
#%% Access images and classify fovea
def classify_fovea(desktop_folder):
    # Import model
    model_path = "./models/fovea_class_100_93_Jan31.h5"
    
    # Load the model
    fovea_classifier = load_model(model_path)
    
    # Testing name
    # desktop_folder = 'Desktop Images'
    # Get filenames of everything in the selected directory
    image_filenames = os.listdir(f"C:/Users/jorda/Desktop/{desktop_folder}")
    
    # Variables to store consecutive fovea predictions
    consecutive_fovea = 0
    start_index = 0
    series = []  # List of tuples (start_index, end_index, length)
    
    # read in images and store in array
    # image_list = []
    for idx, f in enumerate(image_filenames):
        img = cv2.imread(f"C:/Users/jorda/Desktop/{desktop_folder}/{f}")
        # Perform operations to images if needed (Resizing, Denoising, etc.)
        resize = tf.image.resize(img, (205,320))
        # np.expand_dims(resize, 0)
        yhat = fovea_classifier.predict(np.expand_dims(resize/255, 0))
        
        # If fovea is predicted
        if yhat <= 0.7:
            if consecutive_fovea == 0:
                start_index = idx
            consecutive_fovea += 1
        # If no fovea is predicted or it's the last image
        elif consecutive_fovea > 0 or idx == len(image_filenames) - 1:
            series.append((start_index, idx-1, consecutive_fovea))
            consecutive_fovea = 0
        
        # Identify longest series
    if not series:  # If the list is empty (no fovea detected in any image)
        return None
    
    # Determine the longest streak of images containing a fovea
    longest_series = max(series, key=lambda x: x[2])
    central_index = (longest_series[0] + longest_series[1]) // 2
    
    # Return the central image from the longest series
    central_image_filename = image_filenames[central_index]
    central_image = cv2.imread(f"C:/Users/jorda/Desktop/{desktop_folder}/{central_image_filename}")
    
    # Display central_image
    plt.imshow(central_image)
    plt.axis('off')
    plt.title('Central Fovea')
    plt.show()
    
    return central_image_filename
        # # Show fovea prediction images
        # if yhat > 0.7:
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.title(f"No fovea: {100*yhat[0][0]:.2f}%")
        #     plt.show()
        # else:
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.title(f'Fovea: {100*(1-yhat[0][0]):.2f}%')
        #     plt.show()
        #     # Add images with a fovea to image_list
        #     image_list.append(resize)
        # return image_list
central_image_filename = classify_fovea('Desktop Images')

#%% Function to display an image in Spyder using matplotlib
def show_image(title, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    
#%% Green Line Detection
def detect_green_line(central_image_filename):
    # Define the path to the image
    source = f"C:/Users/jorda/Desktop/Desktop Images/{central_image_filename}"
    
    # Read the image
    image = cv2.imread(source)
    # show_image("Original Image",image)
    
    # Convert image to HSV (Hue, Saturation, Value) color space for easier color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # show_image("HSV Image",hsv_image)
    
    # Define the range of green color in HSV
    # These values will need to be adjusted based on the actual green color in the image
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    
# Create a mask to only select the green parts of the image
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Apply the mask to the original image
    green_only = cv2.bitwise_and(hsv_image, hsv_image, mask=green_mask)
    
    # Convert the masked image to grayscale
    gray_image = cv2.cvtColor(green_only, cv2.COLOR_RGB2GRAY)
    
    # Apply edge detection to the grayscale image
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # List to hold all the horizontal lines
    horizontal_lines = []
    
    # Check if any lines were found
    if lines is not None:
        # Go through all the lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the length of the line
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Check if the line is horizontal
            if abs(y2 - y1) < 10:  # Tolerance level for horizontal line, can be adjusted
                horizontal_lines.append((length, x1, y1, x2, y2))
    
    # Sort the lines by length
    horizontal_lines.sort(key=lambda x: x[0], reverse=True)
    
    # Select the middle line out of the three longest lines if there are at least three lines
    middle_line = None
    if len(horizontal_lines) >= 3:
        # The middle line is the second one in the sorted list (since index starts at 0)
        middle_line = horizontal_lines[2][1:]
    elif len(horizontal_lines) > 0:
        # If there are fewer than three lines, select the second longest available line
        middle_line = horizontal_lines[0][1:]
    
    # If a middle line is found, draw it on the image
    if middle_line is not None:
        cv2.line(image, (middle_line[0], middle_line[1]), (middle_line[2], middle_line[3]), (255, 255, 0), 2)
    
    # Show the image using the provided show_image function
    show_image('Detected Line', image)
    
    # Return the start and stop positions of the middle line
    middle_line_coords = middle_line if middle_line else (0, 0, 0, 0)
    
    return middle_line_coords

middle_line_coords = detect_green_line(central_image_filename)
print(middle_line_coords)

#%% Locate Fovea in Classified Images
def locate_fovea(central_image_filename):
    # Load a pretrained YOLOv8n model
    model = YOLO('./models/fovea_location_Feb20.pt')  # pretrained YOLOv8n model
    
    # Define path to the image file
    source = f"C:/Users/jorda/Desktop/Desktop Images/{central_image_filename}"
    
    # Run inference on the source
    results = model.predict(source)  # list of Results objects
    
    # Show the results
    for result in results:
        if result.boxes.cpu().numpy().xyxy.shape[0] != 0: # Check if the fovea was found
            im_array = result.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            
            boxes = result.boxes.cpu().numpy() # Obtain all results
            for box in boxes:
                r = box.xyxy[0].astype(int) # Save bounding box
                center_line = (r[2]-((r[2]-r[0])/2)).astype(int) # Obtain the x coordinate of the fovea
                
                # Display the image and draw a vertical line where center_line is found
                plt.imshow(im)
                plt.axvline(x=center_line, color='r', linestyle='--')
                plt.show()
                print(center_line)
        else:
            print('No fovea found')
    return center_line

center_line = locate_fovea(central_image_filename)

#%% Draw the bounding box on the OCT
def fovea_coordinates(middle_line_coords, center_line, central_image_filename):
    # Extract the bounding box using the adjusted function
    # bbox = extract_enclosed_horizontal_green_line(source)
    x1, y1, x2, y2 = middle_line_coords
    x = x1
    y = y1
    w = x2-x1
    h = y2-y1
    # Define path to the image file
    source = f"C:/Users/jorda/Desktop/Desktop Images/{central_image_filename}"
    
    # Reload the image for visualization
    image = cv2.imread(source)
    oct_image = image.copy()
    
    # Reference Coordinates
    ref_coor = (1000, 600)
    
    # Draw figure
    fovea_x = w-(image.shape[1]-center_line) # x coordinates for fovea
    # Complete fovea coordinates
    fovea_coor = (x+fovea_x, y+h) 
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.rectangle(image, (x, y+h-5), (x+w, y+h+5), (255, 0, 0), 2) # Detected Middle Green Line
    cv2.rectangle(image, (fovea_coor[0]-5, fovea_coor[1]-5), (fovea_coor[0]+5, fovea_coor[1]+5), (255, 0, 255), 10) # Detected Fovea Location
    cv2.rectangle(image, (image.shape[1]-w, y+h-5), (image.shape[1], y+h+5), (0, 0, 255), 2) # Ensure the scan length is the same for fundus and OCT images
    cv2.rectangle(image, (ref_coor[0]-5, ref_coor[1]-5), (ref_coor[0]+5, ref_coor[1]+5), (0, 255, 255), 10) # Reference Coordinates
    show_image("Detected Middle Green Line", image)
    
        
    return fovea_coor, oct_image
    
[fovea_coor, oct_image] = fovea_coordinates(middle_line_coords, center_line, central_image_filename)
    
#%% Generate Publication Figure
def create_multi_panel_figure(oct_image, fovea_location):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns for two images
    plt.subplots_adjust(wspace=0)  # Eliminate white space between panels
    
    # Displaying the original OCT image
    ax[0].imshow(oct_image, cmap='gray')
    ax[0].set_title('Original OCT Image')
    ax[0].axis('off')  # Remove the axis
    
    # Displaying the OCT image with fovea located
    ax[1].imshow(oct_image, cmap='gray')  # Assuming the same image is used for demonstration
    circle = plt.Circle(fovea_location, 25, color='red', fill=False)  # Create a circle to represent the fovea location
    ax[1].add_patch(circle)
    circle = plt.Circle(fovea_location, 75, color='purple', fill=False)  # Create a circle to represent the fovea location
    ax[1].add_patch(circle)
    ax[1].set_title('OCT Image with Fovea Located')
    ax[1].axis('off')  # Remove the axis

    plt.show()
    
create_multi_panel_figure(oct_image, fovea_coor)

def create_fovea_location_figure(oct_image, fovea_location):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)  # Single image
    
    # Displaying the OCT image with fovea located
    ax.imshow(oct_image, cmap='gray')  # Assuming the same image is used for demonstration
    circle = plt.Circle(fovea_location, 25, color='red', fill=False)  # Create a circle to represent the fovea location
    ax.add_patch(circle)
    circle = plt.Circle(fovea_location, 75, color='purple', fill=False)  # Create a larger circle to represent the fovea location
    ax.add_patch(circle)
    ax.set_title('Automatic Foveal Location')
    ax.axis('off')  # Remove the axis

    # Save the figure with high resolution
    plt.savefig('fovea_location_figure.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
create_fovea_location_figure(oct_image, fovea_coor)

#%% Function to detect the horizontal green line (Old)
# def extract_enclosed_horizontal_green_line(central_image_filename):
#     # Define path to the image file
#     source = f"C:/Users/jorda/Desktop/Desktop Images/{central_image_filename}"
#     # Load the image
#     image = cv2.imread(source)
#     # show_image("Original Image", image)

#     # Define range for green color in RGB
#     # The values have been adjusted to detect a range of green in RGB color space
#     lower_green = np.array([0, 245, 0])  # This is a darker green in RGB
#     upper_green = np.array([10, 255, 10])  # This is a lighter green in RGB

#     # Create a mask for the green color
#     mask = cv2.inRange(image, lower_green, upper_green)
#     # show_image("Green Color Mask", mask)
    
#     # Detect edges
#     edges = cv2.Canny(mask, 50, 150)
#     # show_image("Edges", edges)

#     # Find contours in the mask
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     image_with_contours = image.copy()
#     cv2.drawContours(image_with_contours, contours, -1, (255, 255, 0), 3)
#     # show_image("Contours", image_with_contours)

#     # Filter out horizontal lines and sort by width
#     horizontal_lines = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = float(w) / h
#         if aspect_ratio > 5:  # The aspect ratio threshold to filter out vertical lines
#             horizontal_lines.append((x, y, w, h))

#     # Sort the horizontal lines by width (longest first)
#     horizontal_lines.sort(key=lambda x: x[2], reverse=True)

#     # Select the middle one of the three longest lines
#     if len(horizontal_lines) >= 3:
#         middle_line = horizontal_lines[1]  # Index 1 is the second longest line
#     elif len(horizontal_lines) > 0:
#         middle_line = horizontal_lines[0]  # Fallback to the longest line if fewer than 3
#     else:
#         print("No horizontal green lines found.")
#         return None

#     # Draw a bounding box around the middle line
#     x, y, w, h = middle_line
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
#     show_image("Detected Horizontal Line", image)

#     return middle_line

# # Replace 'central_image_filename' with the actual image file name when calling the function.
# enclosed_horizontal_line = extract_enclosed_horizontal_green_line(central_image_filename)