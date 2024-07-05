<h3>Image Recognition using a Convolutional Neural Network with the Back-Propagation Algorithm</h3>
<hr>
This program is a web application written in Go that makes extensive use of the html/template package.
Navigate to the C:\Users\your-name\ConvNeuralNetwork2\src\cnn\ directory and issue "go run imagecnn2.go" to
start the Convolutional Neural Network server. In a web browser enter http://127.0.0.1:8080/imageCNN2
in the address bar.  There are two phases of operation:  the training phase and the testing phase.  During the training
phase, examples consisting of images and the desired class are supplied to the network.  The images
are 256x256 pixel PNG image files that are converted to Go image structures and inserted into two-dimensional matrices.
The network itself is a directed graph consisting of an input layer of the images, a hidden layer of 32 feature maps, and
a flattened output layer of nodes. The 32 feature map outputs are 32x32 planar neurons, they are pooled using 2x averaging, and 
serialized into a 32*16*16+1 one-dimensional array.  The feature maps of the network are connected by weighted
links.  The network is fully connected.  This means that every node is connected to its immediately adjacent neighbor node.  The weights are trained
by first propagating the inputs forward, layer by layer, to the output layer of nodes.  The output layer of nodes finds the
difference between the desired and its output and back propagates the errors to the input layer.  The hidden and input layer
weights are assigned “credit” for the errors by using the chain rule of differential calculus.  Each neuron consists of a
linear combiner and an activation function.  This program uses the hyperbolic tangent function to serve as the activation function.
This function is non-linear and differentiable and limits its output to be between -1 and 1.  <b>The purpose of this program is to classify an
image</b>.
The user selects the MLP training parameters:
<li>Epochs</li>
<li>Learning Rate</li>
<br>
<p>
The <i>Learning Rate</i> is between .01 and .00001.  Each <i>Epoch</i> consists of 16 <i>Training Examples</i>.  
One training example is a png image and the desired class (0, 1,…, 15).  There are 16 images and therefore 16 classes.
The images are converted to 256x256 grayscale of 1 or -1 integers that represent the image.
The 1 represents black (grayscale < 128) and the -1 represents white (grayscale > 128).  The PNG image files were produced using Microsoft Paint3D.
</p>
<p>
When the <i>Submit</i> button on the MLP Training Parameters form is clicked, the weights in the network are trained
and the Learning Curve (mean-square error (MSE) vs epoch) is graphed.  As can be seen in the screen shots below, there is significant variance over the ensemble,
but it eventually settles down after about 30 epochs. An epoch is the forward and backward propagation of all the 16 training samples.
</p>
<p>
When the <i>Test</i> link is clicked, 16 examples are supplied to the MLP.  It classifies the images.
The test results are tabulated and the actual images are graphed from the png files that were supplied to the CNN.
It takes some trial-and-error with the MLP Training Parameters to reduce the MSE to zero.  It is possible to a specify a 
more complex MLP than necessary and not get good results.  For example, using more hidden layers, a greater layer depth,
or over training with more examples than necessary may be detrimental to the MLP.  Clicking the <i>Train</i> link starts a new training
phase and the MLP Training Parameters must be entered again.
</p>

<b>Image Recognition Learning Curve, MSE vs Epoch, One Hidden Layer, 32 feature maps, 32x32 neurons average downsampled 2x, flattened output layer
![image](https://github.com/thomasteplick/imageCNN2/assets/117768679/96cc9123-7ace-41ed-9f41-edfeda2274f4)

<b>Image Recognition Test Results, One Hidden Layer, 32 feature maps, 32x32 neurons average downsampled 2x, flattened output layer
![image](https://github.com/thomasteplick/imageCNN2/assets/117768679/fde79714-3507-4266-99cf-9000b59c22c2)

![image](https://github.com/thomasteplick/imageCNN2/assets/117768679/c3d1819c-121f-4c57-b6d3-9c333443bdb2)

![image](https://github.com/thomasteplick/imageCNN2/assets/117768679/ba766870-82f4-4859-aee0-253063b3ab7b)


