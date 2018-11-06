# Udacity: Self-Driving Car Engineer Nanodegree 

---

## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
**Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. ** 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* this **README.md** summarizing the results
* **vedio.mp4** (in `\imgs_videos`) a vedio recording of the vehicle driving autonomously for several laps.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

[NIVIDIA Autonomous Car group's model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) was employed. It works and is not overly complex.

A summary of the model architecture is below:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3        					   	    	|
| Lambda            	| Normalization          						|
| cropping              | Crop image to filter out noise like trees     |
| Convolution        	| 5x5 kernel, 2x2 stride, 24 depth          	|
| Convolution        	| 5x5 kernel, 2x2 stride, 36 depth          	|
| Convolution        	| 5x5 kernel, 2x2 stride, 48 depth          	|
| Convolution        	| 3x3 kernel, 1x1 stride, 64 depth          	|
| Convolution        	| 3x3 kernel, 1x1 stride, 64 depth          	|
| Flatten       		| 												|
| Fully connected		| 100 depth  									|
| Fully connected		| 50 depth  									|
| Fully connected		| 10 depth  									|
| Fully connected		| 1 depth										|

#### 2. Attempts to reduce overfitting in the model

I decided not to modify the NIVIDIA model by applying techniques like Dropout or Max pooling. 

Instead, I kept the training epochs low: only five epochs. The sample data was split into training and validation data. Using 80% as training and 20% as validation. Training samples from both tracks are used to generalize the model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 123).

#### 4. Appropriate training data

Training data was collected by running simulator in training mode using Joysticks, which can provide smoother steering than keyboard. Around 100K total samples were collected. 

Here are some data collection guidelines I used: 

* four to five laps of center lane driving
* one lap of recovery driving from side to side
* two laps of focusing on driving smoothly on curves
* same techniques for both tracks

Besides data collection, images from center, left, right cameras were used. images and steering from center camera were flipped and fitted into training data set.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use successive refinement of a 'good' model. First, a preliminary model was built, then use that previous model as a starting point for next training session. Generate some new data and 'fine tune' the model.

My first step was to use a convolution neural network model similar to the LeNet with data pre-processing, I thought this model might be appropriate because it has complexity and performs good on image classification. I trained the model with three epochs, the car drove off-road and ran into rocks.

Then, as a second step, a more powerful network-NIVIDA Network was used. New layers was added. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that both the training error and validation error keep decreasing, so I increased the epoch to 5.

To combat the overfitting, I augmented the data by adding the same image flipped with a negative angle. In addition to that, the left and right camera images where introduced with a correction factor on the angle to help the car go back to the lane. The car was able to finish most of the track, but went off road at the last curve where there is crossing of mud road and track. 

Then I collected more data at that specific curve. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network as described previously. Here is a visualization of the architecture:


![](imgs_videos\image1.JPG)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![](imgs_videos\center_camera.JPG)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to center when it starts to leading toward side. These images show what a recovery looks like starting from side:



![center_camera](imgs_videos\center_camera.JPG)
![left camera](imgs_videos\left_camera.JPG)
![right camera](imgs_videos\right_camera.JPG)



Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would avoid the vehicle steering to left too hard.

After the collection process, I had 116577 number of data points. I then preprocessed this data by normalization and cropping.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by

![loss vs epoch](imgs_videos\loss.png)

I used an adam optimizer so that manually training the learning rate wasn't necessary.
