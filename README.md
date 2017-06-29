# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Visualization"
[image2]: ./examples/pre_process_demo.png "Grayscaling"
[image3]: ./examples/augmented_images_demo.png "Augmented"
[image4]: ./project/20kmph.jpg "Traffic Sign 1"
[image5]: ./project/no_entry.jpg "Traffic Sign 2"
[image6]: ./project/no_truck_passing.jpg "Traffic Sign 3"
[image7]: ./project/right_turn.jpg "Traffic Sign 4"
[image8]: ./project/stop.jpg "Traffic Sign 5"
[image9]: ./examples/sample_of_same_type.png "Same Class"
[image10]: ./examples/random_each_class.png "Different Classes"
[image20full]: ./examples/20_km_combined.png "20 km Class"
[imagerightturnfull]: ./examples/right_turn_combined.png "Right Turn Class"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/stepanb2/ml-traffic-sign/blob/master/project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is - 34799
* The size of the validation set is 4410
* The size of test set is - 12630
* The shape of a traffic sign image is - (32, 32, 3)
* The number of unique classes/labels in the data set is - 43

Initial training to validation dataset size ratio was near 12% and after augmentation of images in training dataset it became less than 3%. Recommended ratio is 20% and training/validation dataset re-shuffling might be used to improve results.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a chart showing how the data samples are distributed between different sign classes in each of training, validation and testing dataset.

We can discover from the chart that samples are not equally distributed between classes. And some classes have 10x more entries than others. But percentage of samples per each class is close in all three data sets and might reflect the real world distribution of the signs on the road.

![alt text][image1]

Here is a quick look at randdomly selected images of the each class:

![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to run a basic LeNet on the given training dataset without any data pre processsing. Validation accuracy reached 0.851. After the examination random samples of the same class:

![alt text][image9]

It becames clear that images have been taken at different time of the day and in different lighting conditions. In attempt to unify the input images I've applied [grayscale](https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_grayscale) and [image normalization](https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization) to linearly scales image to have zero mean and unit norm. Also in general feature scaling and normalization improves the convergence speed of stochastic gradient descent algorithms ([Link](https://en.wikipedia.org/wiki/Feature_scaling)). 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

After I've reached validation accuracy of 0.93. I decided to add more data into training dataset by adding two random rotations of the image in range of -15 to 15 degrees and two random shifts of the each image in training dataset.

Here is an example of several original images and four augmented images per each one:

![alt text][image3]

By adding augmented images the training dataset was increased in 5x and same solution reached accuracy of 0.97 of validation dataset and 0.957 on testing dataset.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        			| 
|:---------------------:|:---------------------------------------------:| 
| Input         			| 32x32x3 RGB image   				| 
| Normalization + Grayscale Conversion	| Outputs 32x32x1				|	
| Layer 1: Convolution 5x5 + RELU    	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Max pooling	      			| 2x2 stride,  outputs 14x14x32 		|
| Layer 2: Convolution 5x5 + RELU    	| 1x1 stride, valid padding, outputs 10x10x64 	|
| Layer 3: Convolution 3x3 + RELU    	| 1x1 stride, valid padding, outputs 8x8x96	|
| Max pooling	      			| 2x2 stride,  outputs 4x4x96 			|
| Flatten				| Inputs 4x4x96, outpus 1536			|
| Fully connected + RELU		| Outputs 720       				|
| Fully connected + RELU		| Outputs 480       				|
| Fully connected + RELU		| Outputs 120       				|
| Fully connected + RELU		| Outputs 84       				|
| Dropout 50%				| 						|
| Fully connected			| Outputs 43					|
| Softmax				|         					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

 * Optimizer - [Adam](https://arxiv.org/abs/1412.6980). 
 * The selected batch size - 128
 * Epochs number - 50
 * Learning rate - 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of o.978
* test set accuracy of 0.966

I've started with the base LeNet architecture to train the model and initial results were:
* validation test accuracy of 0.851 and test set accuracy of 0.862

Here is a list of iterations that led to the final model. In general, in case of under-fitting I was trying to increase the number of parameters in the model, by adding new layers and extending existing ones. In case of of over-fitting I was adding new max-pool and/or drop out layers. And as the end target I was keeping  [NVidia ConvNet](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in mind. And was adding more convolution  and fully connected layers.

| Change Description         		| Validation Accuracy		| Testing Accuracy	|
|:-------------------------------------:|:-----------------------------:|:---------------------:|
| Base Lenet  | 0.851   | 0.862 |
| Add Normalization + grayscale transformation  | 0.896   | 0.881 |
| Kernel's depths increased to 24 and 48, + 1 fully connected layer added | 0.937   | 0.915 |
| Kernel's depths increased to  32 and 64, + 1 fully connected layer added  | 0.935   | 0.922 |
| Added rotated and shifted images x3 num of images to original   | 0.966   | 0.954 |
| Added rotated and shifted images x5 num of images to original   | 0.973   | 0.957 |
| A dropout layer  added  | 0.965  |   0.96 |
| Learning rate divind by 2, epochs number increased from 10 to 70  | 0.975  |   0.96 |
| Add third conv layer with 3x3x96 kernel | 0.978  |   0.966 |


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

| Image                    | Description and Code           |
|:------------------------:|:------------------------------:|
| ![alt text][image5] |No entry (17)|
| ![alt text][image7] |Turn right ahead (33)|
| ![alt text][image4] |Speed limit (20km/h) (0)|
| ![alt text][image6] |No passing for vehicles over 3.5 metric tons (10)|
| ![alt text][image8] |Stop (14)|


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| Turn right ahead     			| Yield										|
| Speed limit (20km/h) | Slippery road |
| No passing for vehicles over 3.5 metric tons	      		| No passing for vehicles over 3.5 metric tons					 				|
| Stop		| Stop      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This score is way lower than a test accuracy of 0.966 of the final model.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a No Entry sign (probability of almost 1.0), and that's correct.

The top five soft max probabilities were


| Probability         	|     Prediction	        					| Input Image: ![alt text][image5] |
|:---------------------:|:---------------------------------------------:|:------:| 
| 1.00         			| No Entry (17)   									| |
| 4.43496142e-19     				|  Pedestrians (27) 										||
| 4.03251337e-23					| Dangerous curve to the right 
| 2.18457800e-28	      			| Right-of-way at the next intersection (11)					 				||
| 3.85442388e-35				    | Children crossing (28)      							||


For the second image, the model has failed properly recognise the right turn ahead sign.

 The top five soft max probabilities were

| Probability         	|     Prediction	        					| Input Image: ![alt text][image7] |
|:---------------------:|:---------------------------------------------:|:------:| 
| 0.427        			| Yield (13)   									| |
| 0.113     				| Right-of-way at the next intersection (11) 										||
| 0.103					| Speed limit (120km/h) (8)											||
| 0.09	      			| Speed limit (30km/h) (1)					 				||
| 0.07				    | Priority road (12)      							||

More detailed analysis of the sample images for the "Turn right ahead" shows, that used image is sligtly different (e.x. bolder arrow) from training images for the same sign class.
![alt text][imagerightturnfull]

For the third image,  the model has failed properly recognise the "Speed limit (20km/h)" sign.

The top five soft max probabilities were


| Probability         	|     Prediction	        					| Input Image: ![alt text][image4] |
|:---------------------:|:---------------------------------------------:|:------:| 
| 0.62         			| Slippery road (23)   									| |
| 0.22     				| End of all speed and passing limits (32) 										||
| 0.04					| Double curve (21)											||
| 0.03	      			| Dangerous curve to the right (20)					 				||
| 0.01				    | Beware of ice/snow (30)      							||

Here is an example for training images for "Speed limit (20km/h)" sign class. It looks like the input image is cropped too much comparing to training images. Augmenting data with randomly cropping can improve performance of the model
![alt text][image20full]

For the fourth image, the model is pretty sure that this is a "No passing for vehicles over 3.5 metric tons" sign (probability of almost 1.0) and that's correct.

 The top five soft max probabilities were

| Probability         	|     Prediction	        					| Input Image: ![alt text][image6] |
|:---------------------:|:---------------------------------------------:|:------:| 
| 1.00         			| No passing for vehicles over 3.5 metric tons (10)   									| |
| 3.11396407e-19     				| No passing(9) 										||
| 1.89410789e-23					| Road work (25)											||
| 7.83543798e-26	      			| Speed limit (100km/h) (7)					 				||
| 4.05546181e-26				    | Speed limit (80km/h) (5)      							||

For the fifth image, the model is pretty sure that this is a Stop sign (probability of almost 1.0) and that's correct.

 The top five soft max probabilities were

| Probability         	|     Prediction	        					| Input Image: ![alt text][image8] |
|:---------------------:|:---------------------------------------------:|:------:| 
| 1.00         			| Stop (14)   									| |
| 0.00     				| Speed limit (20km/h) (0) 										||
| 0.00					| Speed limit (30km/h) (1)											||
| 0.00	      			| Speed limit (50km/h) (2)					 				||
| 0.00				    | Speed limit (60km/h) (3)      							||

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

TODO
