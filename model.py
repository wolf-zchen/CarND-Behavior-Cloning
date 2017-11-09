import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Flatten, Dense, Convolution2D
# from keras.layers.pooling import MaxPooling2D

def load_data_from_csv(data_path, skipHeader):
    """
    Load training data from driving_log.csv 
    and split into training and validation 
    """
    samples = []
    with open(data_path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        # if data has header
        if skipHeader:
            next(reader, None)
        for sample in reader:
            samples.append(sample)
    
    train_samples, valid_samples = train_test_split(samples, test_size = 0.2)
    
    return train_samples, valid_samples  
    

def generator(samples, data_path, correction, batch_size):
    """
    Generate required images and angle measurements for training
    samples is list of pairs(3x'image_path', 4x'measurements')
    data_path created for easy train of local machine and AWS
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_img = []
            steering_angles = []
            
            for batch_sample in batch_samples:
                
                # Create path for images from center, left and right cameras
                center_path = data_path+'/IMG/'+batch_sample[0].split('\\')[-1]
                left_path = data_path+'/IMG/'+batch_sample[1].split('\\')[-1]
                right_path = data_path+'/IMG/'+batch_sample[2].split('\\')[-1]
                
                # Read in images from center, left and right cameras
                center_image = cv2.imread(center_path)
                center_image_flip = np.fliplr(center_image) # Filp center image
                left_image = cv2.imread(left_path)
                right_image = cv2.imread(right_path)
                
                # Read center steering and create adjusted steering measurements for the side camera images
                steering_center = float(batch_sample[3])
                steering_center_flip = -steering_center #  Taking the opposite sign of the steering measurement
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                
                # Add images and steering angles to data set
                car_img.extend([center_image, center_image_flip, left_image, right_image])
                steering_angles.extend([steering_center, steering_center_flip, steering_left, steering_right])

            # trim image to only see section with road
            X_train = np.array(car_img)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

def NVIDIAModel():
    """
    Model based on NVIDIA architecture
    """
    ch, row, col = 3, 160, 320  # camera format
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, 
                     input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

"""
Main
"""
# Define data_path
data_path = 'data'

# Load training data
train_samples, validation_samples = load_data_from_csv(data_path, False)

correction = 0.2 # Parameter for left/right steering
batch_size = 128
epoch = 5

# Compile and train the model using the generator function
train_generator = generator(train_samples, data_path, correction, batch_size)
validation_generator = generator(validation_samples, data_path, correction, batch_size)

# Create model
model = NVIDIAModel()

# Train the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=epoch, verbose=1)

model.save('model.h5')

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('myfig')
