---

**Vehicle Detection Project**

[//]: # "Image References"
[image1]: ./examples/car_not_car.png
[hog_image1]: ./examples/hog_example.jpg
[bounding_box_image]: ./examples/bounding_boxes.jpg
[sliding_windows_image]: ./examples/sliding_windows.jpg
[heatmap_image]: ./examples/heatmap.jpg
[video_image]: ./examples/video_image.jpg
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from training images

Extracting all features are to be found in `featuresextractor.py`. Specifically HOG is done between lines #32 and #54.   

To figure out the final parameters I explored with different color spaces, orientations and pixel per cells. I trained the model with a couple different settings and checked how some random images look.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][hog_image1]

####2. Choice of HOG parameters

When choosing the HOG parameters I tried to balance speed of the feature generation, the overlap possible when doing hog sub-sampling, and also the prediction accuracy of the models. I.e. choosing pixel per cells below 8 (e.g. 4) would increase the features extraction time by a lot wouldn't really change the prediction accuracy, and even though it's possible to have a higher percentage overlap between search windows, it won't really help the detection much.

In case of the choice of color space, I tried RGB, HSV, HLS and LUV. RGB was the worth, all others seemed to work quite well. LUV was the best. Also, when deciding which channels to use, there was very little different between using the L channel vs. all three channels. But there still was an ~1 percent increase in the accuracy, so I ended up using all channels.

Looking at the example image above, it's kind of easy to see, that the L channel has most of the information for HOG, but other channels provide some information too, even if sometimes it is having almost no HOG information.

#### 3. Color histogram and spatial binning

Using HOG alone made the model have an accuracy of ~98%. That's already quite good, but having better accuracy helps deal with false positives, so I decided to add features from color histogram and spatially binned image. Adding these increased the model accuracy by about ~1%.

The code for extracting these features can be found in the `featuredextractor.py` between line #21 and #30

####4. Training the classifier

Training the classifier is done in `train.py`, lines #40 to #93. The steps of these are:

1. Find all the images for vehicles and non-vehicles (lines #25 to #37 `train.py`)
2. Load all of them and extract features using `featuresextractor.py`
3. Fit a scaler and normalize all the features
4. Do a random 80/20 split for training and test data.
5. Train the model using LogisticRegression
6. Return both the model and the scaler, so that it can be used while preparing data and prediting

##### Notes on the decisions

Although some of the images contain time series data, and doing a random split can skew the accuracy so that it looks better, the model is simple enough, that I deemed this an acceptable solution. The reported accuracy of the model has to be taken with a grain of salt though (it's 99.3%).

LogisticRegression has been chosen instead of LinearSVM, because they are very similar, but LogisticRegression uses log loss function and also reports proper probabilities, which I found working with easier. Further, based on my tests the accuracy of the two models are very close to each other.

###Sliding Window Search

####1. Sliding windows vs HOG sub-sampling

I tried 3 different approaches with sliding window search.

1. Simple sliding window generation with 4 sizes and 90% overlap, combined with separate HOG generation for the window contents (`sliding_window.py` and `featuresextractor.py` lines #111 to #123)
2. Simple sliding window generation with 4 sizes, 90% overlap, combined with HOG sub-sampling (`sliding_window.py` and `featuresextractor.py` lines #56 to #99)
3. HOG sub-sampling based sliding windows, moving windows by number of HOG cells, and doing HOG sampling using 6 scales (`featuresextractor.py` lines #125 to #200)

From the above 1. was the best performing in the sense, that it was the best at finding cars. However that was the slowest as well, so I moved to doing HOG sub-sampling

Mixing HOG sub-sampling with sliding windows works well enough, but has the problem, that sometimes it uses the same features for 2 windows which are close enough together so that they have the same HOG coordinates. I.e. having a window of size 50 slide with 90% overlap moves it by 5 pixels. HOG sub-sampling on the other hand can only move by 8 pixels, hence there will be windows which have the same HOG and color features.

So after this I moved to moving the windows based on HOG sub-sampling, which with the params I have (8 pixels per cell, 1 cell per step) can do a 87.5% overlap. 

The window scales have been decided by trying a multitude of them. Originally I tried scales from 0.5 to 5.0. However, it became clear, that 0.5 doesn't add anything and 4.0 and 5.0 both mostly just introduced false positives (my assumption is, this happens because compressing the image that much makes areas with shadows look very similar to cars on HOG and maybe for color features as well).

The exact scales were chosen, so that it's a little easier to debug the process in some sense. Also the scale of 1/.65 is used with 2 different regions for better detection (i.e. it would totally miss the white car on some images, once it's a bit farther, and no other scales would find it either). This admittedly is a little bit of a hack, but works well enough.

####2. Finding windows with cars

Ultimately I searched on 6 scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][sliding_windows_image]

![alt text][bounding_box_image]



------



### Video Implementation

####1. Final video

Processing images for the video is done in `vehicles.py` lines #190 to #237

Here's a [link to my video result](./project_video.mp4)


####2. Calculating bounding boxes

From the positive window detections (done in `vehicles.py` lines #19 to #75) I created a heatmap, using the scores returned. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected (`vehicles.py` lines #78 to #114)

The generated heatmap looks like this

![alt text][heatmap_image]

#### 3. Filtering false positives

I used a couple ways to deal with false positives

1. Increasing the minimum probability for a detection to be 85%
2. Combining the heatmaps of the last 15 frames, with decreasing weights the older the frame
3. Thresholding the heatmap so that only areas which has a high confidence remain
4. Dropping bounding boxes which have obviously bad shape (e.g. are too high compared to their width)
5. Hard negative mining and retraining the model with the negative examples
6. Removing scales which didn't provide much to the result, but added too many false detections

To generate the heatmap, I calculated a score for every detection. The score was calculated by linearly scaling the probability from between 0.85-1.0 to between 1-5. This way I could be sure, that a high confidence detection is never removed by thresholding, and that an area has to have more than one low confidence detections to qualify. 

This is especially useful in case of the videos, where subsequent frames very rarely have multiple detections when they are false positive. The result looks like below.

![alt text][video_image]

This also includes the in a pip style the detections, the original heatmap and the thresholded heatmap. The heatmap is thresholded both to minimum and maximum values, so that we can have a better picture of where the bounding boxes are.

---

###Discussion

####1. HOG subsampling missing cars

HOG sub-sampling doesn't always perform well, because based on what's the area of the HOG sub-sampling might miss cars.

One way to deal with these is to have the same scale with different sampling regions (basically shifting the region, so that sub-sampling will cover more windows). Another way would be to train the model, so that it can deal with shifting. This has other benefits as well, but ultimately it's a little more complex to implement, would increase training time, etc.

#### 2. Hard negative mining

Originally it seemed, that thresholding, dropping bounding boxes which have shapes so that they can't be vehicles and increasing the minimum probability for detection is going to be enough. There were however a couple parts in the video, where the vehicles were being detected with very high confidence, even though there was nothing there.

First I scales corresponding to larger window sizes, which got rid of some of these detections. However others remained and no tweaking of thresholding seemed to help. So I decided to get these areas as negative examples a retrain my model. I selected an x range where on the whole video there shouldn't be detections, and whenever a positive result was returned, I saved it. Then I retrained my model including these examples (~70 images), and finally got rid of almost all the false positives. The ones remaining even if they got through the detection threshold, didn't get through the heatmap thresholding

### Possible improvements

#### 1. Using CNN instead of a Linear model and HOG

Using a CNN instead of a linear model with HOG would possibly speed up the detection, even if using sliding windows. This would also make the pipeline more reliable, as CNNs are better fit to classify images. Training speed would probably suffer, but that's not a high price for faster detection.

#### 2. Use SSD or YOLO instead of sliding window search

Implementing a faster CNN based solution instead of a sliding window search could potentially speed up the detection of bounding boxes to real time. While I researched this, I ultimately decided against doing it, because I didn't want to simply reuse a CNN structure fit for general object detection as opposed to classifying cars and not cars, and implementing one from scratch seemed to be more time consuming then doing the linear model + sliding window search. Also there's a lot to learn from using simple models instead of a very fine grained solution.

