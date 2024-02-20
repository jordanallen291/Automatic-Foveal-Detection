#%% 1. Setup and Load Data
#%% Install Dependencies and Setup
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf 
from tensorflow.keras.models import load_model
import os
import cv2
import sys
print(tf.config.list_physical_devices('GPU'))

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
# For every gpu utilized, limit memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
        
#%% Load Data

# Define the directories for each class
fovea_dir = 'data/fovea'
no_fovea_dir = 'data/no fovea'

# Get a list of file paths for each class
fovea_files = [os.path.join(fovea_dir, f) for f in os.listdir(fovea_dir)]
no_fovea_files = [os.path.join(no_fovea_dir, f) for f in os.listdir(no_fovea_dir)]

# Randomly sample from the no_fovea_files to match the number of fovea_files
no_fovea_files = random.sample(no_fovea_files, len(fovea_files))

# Define the selected directory
selected_dir = 'selected data'

# # If the selected directory already exists, remove it and all its contents
# if os.path.exists(selected_dir):
#     shutil.rmtree(selected_dir)

# # Create the selected directory
# os.makedirs(selected_dir, exist_ok=True)

# # Create sub-folders for each class
# selected_fovea_dir = os.path.join(selected_dir, 'fovea')
# selected_no_fovea_dir = os.path.join(selected_dir, 'no fovea')
# os.makedirs(selected_fovea_dir, exist_ok=True)
# os.makedirs(selected_no_fovea_dir, exist_ok=True)

# # Copy the selected files to the new directories
# for f in tqdm(fovea_files):
#     shutil.copy(f, selected_fovea_dir)
# for f in tqdm(no_fovea_files):
#     shutil.copy(f, selected_no_fovea_dir)

# Create a dataset from the file paths and labels
desired_height, desired_width = 205, 320  # Original size is 1636 by 2560
data = tf.keras.utils.image_dataset_from_directory('selected data', label_mode='binary', image_size = (desired_height, desired_width), batch_size = 32)
# Allow access to pipeline
data_iterator = data.as_numpy_iterator()
# Get another batch from iterator
batch = data_iterator.next()

#%% Preview Batch Images

# Class 0 = fovea
# Class 1 = no fovea

# View a set of images
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
# Flatten the 2x2 grid of axes for easier indexing
ax = axs.ravel()
# Set figure titles
classification = ['fovea','no fovea']
for idx, img in enumerate(batch[0][:4]):
    img = img / 255.0  # Normalize the image data (if not already done)
    ax[idx].imshow(img)
    label = int(batch[1][idx][0])  # Extract scalar value from numpy array
    ax[idx].title.set_text(classification[label])
    ax[idx].axis('off')  # Turn off the axis tick marks

plt.tight_layout()  # Adjust spacing between subplots for better layout
plt.show()


#%% 2. Preprocess Data
#%% Scale Data
# Apply data transformation to data pipeline
def augment_image(image, label):
    # Random brightness (up to 10% of original value)
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast (up to 20% of original value)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random saturation (up to 20% of original value)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Random hue (up to 10% of original value)
    image = tf.image.random_hue(image, max_delta=0.1)
    
    return image, label

# Apply scaling and data augmentation transformations to the data pipeline
data = data.map(lambda x, y: (x / 255, y))
data = data.map(augment_image)

# Continue with the rest of the processing
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()



#%% View as a set of images

# Prepare figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
# Flatten the 2x2 grid of axes for easier indexing
ax = axs.ravel()
# Set figure titles
classification = ['fovea','no fovea']
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    label = int(batch[1][idx][0])  # Extract scalar value from numpy array
    ax[idx].title.set_text(classification[label])
    ax[idx].axis('off')  # Turn off the axis tick marks

plt.tight_layout()  # Adjust spacing between subplots for better layout
plt.show()

    
#%% Split Data
train_size = int(len(data)*.7)
val_size = int((len(data)-train_size)/2)
test_size = int((len(data)-train_size)/2)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#%% 3. Deep Model
#%% Build Deep Learning Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

# Setup Model
model = Sequential()

# Add Convolutional and MaxPooling layers

# First Convolutional block
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(desired_height, desired_width, 3)))
model.add(MaxPooling2D())

# Second Convolutional block with increased filters
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())

# Third Convolutional block with even more filters
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())

# Fourth Convolutional block
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())

# Flattens to a single value
model.add(Flatten())

# Adding more Dense layers to improve the capacity of the network
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))

# Final output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

#%% Train
# Define the path to save the best model
checkpoint_path = "models/best_model.ckpt"

# Create a callback that saves the best model's weights based on validation accuracy
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',  # Monitor validation accuracy
    mode='max',              # Mode is set to 'max' to monitor accuracy
    verbose=1,
    save_best_only=True,     # Only save the model with highest validation accuracy
    save_weights_only=True   # Save only weights (if you want the full model, set this to False)
)

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback, cp_callback])

#%% Plot Performance

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#%% Plot Accuracy

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#%% 4. Evaluate Performance
#%% Load Best Model
model.load_weights(checkpoint_path)
# Evaluate
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X, verbose=0)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
    
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

#%% Prepare test data

# Directories
fovea_dir = 'C:/Users/jorda/OneDrive - University of Utah/Personal/Work/Moran Eye Center/Masters/data/classification test images/fovea'
no_fovea_dir = 'C:/Users/jorda/OneDrive - University of Utah/Personal/Work/Moran Eye Center/Masters/data/classification test images/no fovea'

# Collect all image file paths
fovea_images = [os.path.join(fovea_dir, f) for f in os.listdir(fovea_dir) if f.endswith('.jpg')]
no_fovea_images = [os.path.join(no_fovea_dir, f) for f in os.listdir(no_fovea_dir) if f.endswith('.jpg')]

# Randomly sample from the no_fovea_files to match the number of fovea_files
no_fovea_images = random.sample(no_fovea_images, len(fovea_images))

#%% Test the resulting model

# Prepare counter
count = 1
# Show every nth image
nth = 5
# Correct fovea count
correct_fovea = 0
# Test each image
for img_path in fovea_images:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resize = tf.image.resize(img, (desired_height, desired_width))
    
    # Predict
    yhat = model.predict(np.expand_dims(resize/255, 0), verbose=0)
    if yhat < 0.7:
        correct_fovea += 1
    # print(f"Prediction Value: {yhat[0][0]}")
    if count % nth == 0 :
        # Interpret prediction
        if yhat < 0.7:
            title = (f'Fovea: {100*(1-yhat[0][0]):.2f}%')
        else:
            title = (f"No fovea probability: {100*yhat[0][0]:.2f}%")
        # Show the image
        plt.imshow(resize.numpy().astype(int))
        plt.title(title)
        plt.show()
    count += 1

accuracy = correct_fovea/len(fovea_images)
print(f'Achieved {100*accuracy:.2f}% accuracy')

# Prepare counter
count = 1
# Show every nth image
nth = 5
# Correct non-fovea count
correct_non_fovea = 0
# Test each image
for img_path in no_fovea_images:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resize = tf.image.resize(img, (desired_height, desired_width))
    
    # Predict
    yhat = model.predict(np.expand_dims(resize/255, 0),verbose = 0)
    if yhat >= 0.7:
        correct_non_fovea += 1
    # print(f"Prediction Value: {yhat[0][0]}")
    if count % nth == 0 :
        # Interpret prediction
        if yhat < 0.7:
            title = (f'Fovea: {100*(1-yhat[0][0]):.2f}%')
        else:
            title = (f"No fovea probability: {100*yhat[0][0]:.2f}%")
        # Show the image
        plt.imshow(resize.numpy().astype(int))
        plt.title(title)
        plt.show()
    count += 1

accuracy = correct_non_fovea/len(no_fovea_images)
print(f'Achieved {100*accuracy:.2f}% accuracy')
    
#%% 5. Save the Model
#%% Save the model

save_model = input('Would you like to save this model?\n(yes/no): ')
if save_model.lower() == 'yes':
    model_name =input('Please enter a model_name:\n')
    model.save(os.path.join('models',f'{model_name}.h5'))

    #%% Load new model
    new_model = load_model(os.path.join('models',f'{model_name}.h5'))
    
    #%% Test the model
    
    # Prepare counter
    count = 1
    # Show every nth image
    nth = 5
    # Correct fovea count
    correct_fovea = 0
    # Test each image
    for img_path in fovea_images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        resize = tf.image.resize(img, (desired_height, desired_width))
        
        # Predict
        yhat = new_model.predict(np.expand_dims(resize/255, 0),verbose = 0)
        if yhat < 0.7:
            correct_fovea += 1
        # print(f"Prediction Value: {yhat[0][0]}")
        if count % nth == 0 :
            # Interpret prediction
            if yhat < 0.7:
                title = (f'Fovea: {100*(1-yhat[0][0]):.2f}%')
            else:
                title = (f"No fovea probability: {100*yhat[0][0]:.2f}%")
            # Show the image
            plt.imshow(resize.numpy().astype(int))
            plt.title(title)
            plt.show()
        count += 1
    
    accuracy = correct_fovea/len(fovea_images)
    print(f'Achieved {100*accuracy:.2f}% accuracy')
    
    # Prepare counter
    count = 1
    # Show every nth image
    nth = 5
    # Correct non-fovea count
    correct_non_fovea = 0
    # Test each image
    for img_path in no_fovea_images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        resize = tf.image.resize(img, (desired_height, desired_width))
        
        # Predict
        yhat = new_model.predict(np.expand_dims(resize/255, 0),verbose = 0)
        if yhat >= 0.7:
            correct_non_fovea += 1
        if count % nth == 0 :
            # Interpret prediction
            if yhat < 0.7:
                title = (f'Fovea: {100*(1-yhat[0][0]):.2f}%')
            else:
                title = (f"No fovea probability: {100*yhat[0][0]:.2f}%")
            # Show the image
            plt.imshow(resize.numpy().astype(int))
            plt.title(title)
            plt.show()
        count += 1
    
    accuracy = correct_non_fovea/len(no_fovea_images)
    print(f'Achieved {100*accuracy:.2f}% accuracy')
