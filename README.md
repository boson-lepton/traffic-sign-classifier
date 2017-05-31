# Traffic Sign Classifier


Traffic sign classifier learning architecture is based on the LeNet-5 neural network architecture.

## Architecture
**Layer 1: Convolution** with strides=1 and output shape 28x28x12. **relu activation** and **pooling** output 14x14x12.

**Layer 2: Convolution** with strides=1 and output shape 10x10x32. **relu activation** and **pooling** output 5x5x32 and **flatten** shape 800 outputs

**Layer 3: Fully Connected** with 240 outputs and **relu activation**

**Layer 4: Fully Connected** with 168 outputs, **relu activation** and **dropout**

**Layer 5: Fully Connected** with 43 (n_classes) outputs

### Convolution
The primary purpose of Convolution is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data.
A filter slides over the input image (convolution operation) to produce a feature map. The convolution of another filter, over the same image gives a different feature map.

### Activation Function (ReLU)
ReLU is an element wise operation (applied per pixel) which replaces all negative pixel values in the feature map by zero. The purpose of ReLU is to introduce non-linearity in our network, since most of the real-world data we would want our network to learn would be non-linear (Convolution is a linear operation – element wise matrix multiplication and addition, so we account for non-linearity by introducing a non-linear function like ReLU).
One major benefit is the reduced likelihood of the gradient to vanish. This arises when input > 0. In this regime the gradient has a constant value. In contrast, the gradient of sigmoids becomes increasingly small as the absolute value of x increases. The constant gradient of ReLUs results in faster learning.
The other benefit of ReLUs is sparsity. Sparsity arises when input <= 0. The more such units that exist in a layer the more sparse the resulting representation. Sigmoids on the other hand are always likely to generate some non-zero value resulting in dense representations. Sparse representations seem to be more beneficial than dense representations.

### Pooling
Architectures with a max pooling operation converge considerably faster than those employing  subsampling operation. Furthermore, they seem to be superior in selecting invariant features and improve generalization.

### Droput
Droput value of 0.25 was empirically choosen to give maximum accuracy over 30 epochs.

### Loss function
The cross-entropy cost function has the benefit that, unlike the quadratic cost, it avoids the problem of learning slowing down.

## Input
The input is a 32x32xC with C = 3 color channels.

## Ouput
Returns the results of Layer 5


## Model parameters
The adam optimizer was used with a learning rate of 0.001. Batchsize was 1000 with 30 epochs and dropout rate of .25. Weights were initialised with truncated_normal and mu = 0, sigma = 0.1.

### Training and testing

EPOCH   1 Validation accuracy = 0.491
EPOCH   2 Validation accuracy = 0.645
EPOCH   3 Validation accuracy = 0.740
EPOCH   4 Validation accuracy = 0.795
EPOCH   5 Validation accuracy = 0.833
EPOCH   6 Validation accuracy = 0.871
EPOCH   7 Validation accuracy = 0.891
EPOCH   8 Validation accuracy = 0.909
Adding run metadata for  1000
EPOCH   9 Validation accuracy = 0.922
EPOCH  10 Validation accuracy = 0.931
EPOCH  11 Validation accuracy = 0.939
EPOCH  12 Validation accuracy = 0.940
EPOCH  13 Validation accuracy = 0.950
EPOCH  14 Validation accuracy = 0.952
EPOCH  15 Validation accuracy = 0.953
EPOCH  16 Validation accuracy = 0.961
Adding run metadata for  2000
EPOCH  17 Validation accuracy = 0.961
EPOCH  18 Validation accuracy = 0.963
EPOCH  19 Validation accuracy = 0.965
EPOCH  20 Validation accuracy = 0.967
EPOCH  21 Validation accuracy = 0.973
EPOCH  22 Validation accuracy = 0.971
EPOCH  23 Validation accuracy = 0.971
EPOCH  24 Validation accuracy = 0.971
Adding run metadata for  3000
EPOCH  25 Validation accuracy = 0.976
EPOCH  26 Validation accuracy = 0.975
EPOCH  27 Validation accuracy = 0.974
EPOCH  28 Validation accuracy = 0.981
EPOCH  29 Validation accuracy = 0.978
EPOCH  30 Validation accuracy = 0.982
Test accuracy = 0.919

Validation accuracy after 30 epochs is 0.982 and test accuracy is 0.919


## Data Preprocessing and Augmentation
Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization.
The normType used is MINMAX such that the minimum value is alpha=0.1 and maximum value is beta=0.9.
Histogram equalization is used to improve the contrast of input images.
Above techniques improve classification predictions.

Images were converted and split into YUV components. Y was equalized before merging back. This made the image brightness more consistent. The colourspace was converted back to RGB (YUV images didnt plot well) as it appeared to make no difference to accuracy when training.

Jittered images were generated by rotating randomly between [-20,-15,-10,-5,5,10,15,20] degrees with a random perturb in the x and y direction of [0,2] pixels applied.
The jittered data was then split randomily with 90% to train and 10 % to test. This resulted in a training set of 122633 with 13626 for validation test.
The original training set had 39209 examples with some sign categories having few images. The new jittered dataset had a more equal number of training images per category.

y values are one hot encoded as tensorflow variables later.
Test data was not augmented with jittered data.


## Test a Model on New Images
![Test Images](https://github.com/boson-lepton/traffic-sign-classifier/blob/master/test/test1.png?raw=true)

**80 KM sign** The classifier has misclassified this image. based on my understanding, it is beacuse of poor contrast. 

![New Images](https://github.com/boson-lepton/traffic-sign-classifier/blob/master/test/test2.png?raw=true)

**Hand held stop sign** This image has been misclassified because the classfier hasn't seen anything like this during training.
                        The training samples have octagonal shaped stop signs. It's expected, I guess. 

**pedestrian crossing** This image has been misclassified because the classfier hasn't seen anything like this during training.
                        It's expected, I guess.

**50 KM sign**          Classified correctly but not with high confidence.

**Yield sign**          Classified correctly with high confidence. 

**Yield sign**          Misclassified, I'm surprised as to why.



