
# Image Classification
<h4>
<a href="#back"> Background Knowledge</a> | 
<a href="#fil"> Filters </a> | 
<a href="#sub">Subsamping Layers</a> |
<a href="#pad">Padding</a> |
<a href="#res">Output Size</a> |
<a href = "#ref">References</a>
</h4>

<p id="back"> </p>

## Background Knowledge


Convolutional Neural Networks (CNNs), used in image classification, successfully extract features from an image, as shown in Fig.1. In the early layers of the CNNs' architecture, low-level features are extracted. Later layers, or the fully-connected layers, use these features to classify the image and determine a class label. Having multiple CNNs in the early layers produces a combination of low-level features and builds a feature hierarchy. A feature hierarchy is an amalgamation of edges and blobs to form high-level features. High-level features create more complex shapes, such as the general contours of objects like animals, cars, and boats. Therefore, extracting features into a hierarchical feature map results in high performance in image classification and, in general, any image-related tasks. 
| ![CNN.png](https://github.com/luisIvey05/Computer_Vision/blob/main/Image_Classification/images/CNN.png) |
|:--:|
| Fig. 1 A deep CNN from <a href=https://subscription.packtpub.com/book/data/9781801819312/14/ch14lvl1sec98/implementing-a-deep-cnn-using-pytorch> *Machine Learning with PyTorch and Scikit-Learn* </a>|


How CNNs compute feature maps from an input image, shown in Fig. 2, is based on local receptive fields. Local receptive fields or a local patch of pixels from the input image are mapped to a pixel in the output image. Local receptive fields are possible because of sparse connectivity within an image, allowing for parameter sharing. Sparse connectivity is the process of mapping input patches to output pixels. An example of the perks of sparse connectivity is,  translating an image from 5 pixels to the left or 5 pixels to the right would give the same prediction because nearby pixels in images are typically highly correlated. Parameters, or weights, can then be shared among this patch of pixels. Parameter sharing will then allow CNNs to substantially decrease the number of parameters in the network when fed into a fully connected layer. 


![Fig. 2 Local receptive field being mapped to a single pixel from<a href=https://blog.christianperone.com/2017/11/the-effective-receptive-field-on-cnns/l> *Terra Incognita by Christian S. Perone* </a>](https://github.com/luisIvey05/Computer_Vision/blob/main/Image_Classification/images/res.png)


<p id="fil"> </p>

## Filters



<p id="sub"> </p>

## Subsampling Layers


<p id="pad"> </p>

## Padding


<p id="out"> </p>

## Output Size


<p id="ref"> </p>

## References
