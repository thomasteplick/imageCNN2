/*
Neural Network (nn) using convolutional Feature Map architecture.
This is a web application that uses the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/imageCNN2.  There are two phases of
operation:  the training phase and the testing phase.  Epochs consising of
a sequence of examples are used to train the nn.  Each example consists
of a png image and a desired class output.  The nn
itself consists of an input layer of nodes, one or more hidden layers of nodes,
and an output layer of nodes.  The nodes are connected by weighted links.  The
weights are trained by back propagating the output layer errors backward to the
input layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layers.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.

The Convolutional Neural Network cosists of Feature Maps which are two-dimensional
arrays (planes) of neurons that are arranged in layers.  Each Feature Map has a
8x8 filter or kernel (64 weights) that is used to perform a convolution with previous
layer Feature Maps.  The stride or displacement of each convolution is eight.
Padding is used so that the convolution produces the same size output Feature Map as
the input Feature Map.  The input layer is the 256x256 pixel image.  There is
one hidden layer, the hidden layer consists of a convolution operation and
a downsample operation.  The downsample operation reduces the Feature Map width and
height by a factor of two.  The last hidden layer is flattened and fully connected
to the output layer.  The hidden layer Feature Maps depths and sizes are as follows:
32@32x32 (32*8 weights).  Each filter in the Feature Maps has a bias input of 1 and weight.
The output layer also has a bias input of 1 and weight.  The output layer is 4@1X1.
This architecture can classify 2^4 = 16 images.  The flattened fully-connected layer
between the last hidden layer and the output layer is 32*16*16 + 1 = 8193 neurons.
These are fully connected to the output layer consisting of the four neurons.
This will require 4*8193 weights.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	addr               = "127.0.0.1:8080"             // http server listen address
	fileTrainingCNN    = "templates/trainingCNN.html" // html for training CNN
	fileTestingCNN     = "templates/testingCNN.html"  // html for testing CNN
	patternTrainingCNN = "/imageCNN2"                 // http handler for training the CNN
	patternTestingCNN  = "/imageCNN2test"             // http handler for testing the CNN
	xlabels            = 11                           // # labels on x axis
	ylabels            = 11                           // # labels on y axis
	fileweights        = "weights.csv"                // cnn weights
	a                  = 1.7159                       // activation function const
	b                  = 2.0 / 3.0                    // activation function const
	K1                 = b / a
	K2                 = a * a
	dataDir            = "data/"              // directory for the weights and images
	maxClasses         = 40                   // max number of images to classify
	imgWidth           = 256                  // image width in pixels
	imgHeight          = 256                  // image height in pixels
	imageSize          = imgWidth * imgHeight // image size in pixels = 65536
	classes            = 16                   // number of images to classify
	rows               = 300                  // rows in canvas
	cols               = 300                  // columns in canvas
	hiddenLayers       = 1                    // number of hidden layers
	kernelDim          = 8                    // kernel dimension, height and width
	stride             = 8                    // stride for filter forward and backward
)

// Type to contain all the HTML template actions
type PlotT struct {
	Grid         []string  // plotting grid
	Status       string    // status of the plot
	Xlabel       []string  // x-axis labels
	Ylabel       []string  // y-axis labels
	LearningRate string    // size of weight update for each iteration
	Epochs       string    // number of epochs
	TestResults  []Results // tabulated statistics of testing
	TotalCount   string    // Results tabulation
	TotalCorrect string
}

// Type to hold the minimum and maximum data values of the MSE in the Learning Curve
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// graph node
type Node struct {
	y     float64 // output of this node for forward prop
	delta float64 // local gradient for backward prop
}

// filter or kernel used in convolution on the y or delta
// forward prop filters the y, backward prop filters the delta
type Filter struct {
	wgt     [kernelDim][kernelDim]float64 // kernel weights used in convolution for a Feature Map
	biaswgt float64                       // bias weight whose input is constant 1
	layer   int                           // Feature Map (FM) layer
	n       int                           // nth Feature Map in this layer, one filter per FM
}

type Stats struct {
	correct    []int // % correct classifcation
	classCount []int // #samples in each class
}

// training examples
type Sample struct {
	name    string    // image; eg., elephant
	desired int       // numerical class of the image
	image   [][]uint8 // png image 256*256 pixels
}

// Feature Map consists of height, width, and Nodes
type FeatureMap struct {
	h    int // height of Feature Map
	w    int // width of Feature Map
	data [][]Node
}

// Primary data structure for holding the CNN state
type CNN struct {
	plot         *PlotT         // data to be distributed in the HTML template
	Endpoints                   // embedded struct
	link         [][]Filter     // links in the graph which connect the Feature Maps (nodes)
	wgtOutput    []float64      // output flattened weights don't use kernel
	node         [][]Node       // last two flattened layers in the graph are not Feature Maps
	fm           [][]FeatureMap // Feature Maps in the graph, consider them the nodes, [layer, n]
	samples      []Sample
	statistics   Stats
	mse          []float64 // mean square error in output layer per epoch used in Learning Curve
	epochs       int       // number of epochs
	learningRate float64   // learning rate parameter
	desired      []float64 // desired output of the sample
}

// test statistics that are tabulated in HTML
type Results struct {
	Class   string // int
	Correct string // int      percent correct
	Image   string // image
	Count   string // int      number of training examples in the class
}

// global variables and CNN architecture
var (
	// parse and execution of the html templates
	tmplTrainingCNN *template.Template
	tmplTestingCNN  *template.Template
	// number of Feature Maps in the input + hidden layers
	numFMs [hiddenLayers + 1]int = [hiddenLayers + 1]int{1, 32}
	// Two-dimension sizes of the Feature Maps in the layers
	sizeFMs [hiddenLayers + 1]int = [hiddenLayers + 1]int{256, 16}
	// staging location for pooling of convolution
	clipboard [imgHeight][imgWidth]Node
)

// calculateMSE calculates the MSE at the output layer every epoch
func (cnn *CNN) calculateMSE(epoch int) {
	// loop over the output layer nodes
	var err float64 = 0.0
	outputLayer := len(cnn.node) - 1
	for n := 0; n < len(cnn.node[outputLayer]); n++ {
		// Calculate (desired[n] - cnn.node[L][n].y)^2 and store in cnn.mse[n]
		err = float64(cnn.desired[n]) - cnn.node[outputLayer][n].y
		err2 := err * err
		cnn.mse[epoch] += err2
	}
	cnn.mse[epoch] /= float64(classes)

	// calculate min/max mse
	if cnn.mse[epoch] < cnn.ymin {
		cnn.ymin = cnn.mse[epoch]
	}
	if cnn.mse[epoch] > cnn.ymax {
		cnn.ymax = cnn.mse[epoch]
	}
}

// determineClass determines testing example class given sample number and sample
func (cnn *CNN) determineClass(j int, sample Sample) error {
	// At output layer, classify example, increment class count, %correct

	// convert node outputs to the class; 0.5 is the threshold for logistic function
	class := 0
	for i, output := range cnn.node[1] {
		if output.y > 0.5 {
			class |= (1 << i)
		}
	}

	// Assign Stats.correct, Stats.classCount
	cnn.statistics.classCount[sample.desired]++
	if class == sample.desired {
		cnn.statistics.correct[class]++
	}

	return nil
}

// class2desired constructs the desired output from the given class
func (cnn *CNN) class2desired(class int) {
	// tranform int to slice of 0 and 1 representing the 0 and 1 bits
	for i := 0; i < len(cnn.desired); i++ {
		if class&1 == 1 {
			cnn.desired[i] = 1
		} else {
			cnn.desired[i] = 0
		}
		class >>= 1
	}
}

// convolve delta and y to update the 8x8 filter with no padding.
func (cnn *CNN) updateFilter(layer int, i1, i2, d1 int) error {
	// conv(Y, Delta) with no padding; complete overlap of Y and Delta.
	// Multiply Conv(Y,Delta)  by learning rate and add to current filter
	// Upsample the delta in a 2x2 averaging window so it is the same size as Y.

	// height and width of the Y from previous layer and the upsampled delta
	dim := sizeFMs[layer]
	// Use L for rotating 180 deg
	//L := dim - 1

	sum := 0.0
	for row := 0; row < dim/stride; row++ {
		for col := 0; col < dim/stride; col++ {
			// upsample the average delta in a 2x2 window, this is not efficient
			avg := cnn.fm[layer+1][i1].data[row/2][col/2].delta / 4.0
			// Rotate the kernel 180 deg
			//sum += cnn.fm[layer][i2].data[L-row][L-col].y * avg
			sum += cnn.fm[layer][i2].data[row][col].y * avg
		}
	}
	wgtDelta := sum * cnn.learningRate
	depth := i2*d1 + i1
	// update the weights in the kernel with the same convolution
	for j := 0; j < kernelDim; j++ {
		for i := 0; i < kernelDim; i++ {
			cnn.link[layer][depth].wgt[j][i] += wgtDelta
		}
	}

	return nil
}

// Convolve in the backward propagation direction.
// Filter the local gradients from the downstream layer FMs
func (cnn *CNN) filterB(f *Filter, layer, i1 int) error {
	// Convolve the filter over the delta and son't use padding
	// around the edges.

	// Perform the operations in the clipboard
	data := clipboard

	// height and width of the data that the filter convolves over
	dim := sizeFMs[layer]
	// Rotate the filter in x and y coordinates
	L := kernelDim - 1
	for row := 0; row < dim; row += stride {
		for col := 0; col < dim; col += stride {
			sum := 0.0
			curRow := row
			for j := 0; j < kernelDim; j++ {
				curCol := col
				for i := 0; i < kernelDim; i++ {
					//sum += f.wgt[j][i] * cnn.fm[layer][i1].data[curRow][curCol].delta
					sum += f.wgt[L-j][L-i] * cnn.fm[layer][i1].data[curRow][curCol].delta
					curCol++
				}
				curRow++
			}
			// save the filtered delta in clipboard
			data[row/stride][col/stride].delta = sum
		}
	}

	// Put the filtered delta back in the FeatureMap.data[row][col].delta
	for row := 0; row < dim/stride; row++ {
		for col := 0; col < dim/stride; col++ {
			cnn.fm[layer][i1].data[row][col].delta = data[row][col].delta
		}
	}

	return nil
}

// convolve in the forward propagation direction
// filter the fm[][].data[][].y from the previous layer FM
func (cnn *CNN) filterF(f *Filter, layer, i1, i2 int) error {
	// Convolve the filter over the Feature Map and don't use
	// padding around the edges.
	// Include the bias weight in the convolution.
	// Compute activation function phi for the convolution output.
	// Downsample by 2 by finding avg over 2x2 window.
	// Put the downsampled FM.data into this layer FM.data.
	// Process the data in the clipboard.

	// Perform the operations in the clipboard
	data := clipboard

	// height and width of the data that the filter convolves over
	dim := sizeFMs[layer]
	for row := 0; row < dim; row += stride {
		for col := 0; col < dim; col += stride {
			// multiply bias weight by constant 1
			sum := f.biaswgt
			curRow := row

			for j := 0; j < kernelDim; j++ {
				curCol := col
				for i := 0; i < kernelDim; i++ {
					sum += f.wgt[j][i] * cnn.fm[layer][i2].data[curRow][curCol].y
					curCol++
				}
				curRow++
			}
			// compute output y = Phi(v) with the activation function
			data[row/stride][col/stride].y = math.Max(0.0, sum)
		}
	}

	// Process the data in the clipboard.
	// Downsample by 2 by finding avg over 2x2 window.
	// Put the downsampled FM.data into this layer's FM.data.
	step := 2
	for row := 0; row < dim/stride; row += step {
		row2 := row / 2
		for col := 0; col < dim/stride; col += step {
			col2 := col / 2
			sum := 0.0
			for j := 0; j < step; j++ {
				for i := 0; i < step; i++ {
					sum += data[row+j][col+i].y
				}
			}
			// average and propagate to next layer
			cnn.fm[layer+1][i1].data[row2][col2].y = sum / 4.0
		}
	}

	return nil
}

func (cnn *CNN) propagateForward(samp Sample, epoch int) error {
	// Assign sample to input layer FM
	layer := 0
	for j := 0; j < imgHeight; j++ {
		for i := 0; i < imgWidth; i++ {
			cnn.fm[layer][0].data[j][i].y = float64(samp.image[j][i])
		}
	}

	// calculate desired from the class
	cnn.class2desired(samp.desired)

	// Loop over layers: input + hiddenLayers + output layer
	// input->first hidden, then hidden->hidden,..., then hidden->output
	for layer := 1; layer < len(numFMs); layer++ {
		// Loop over FMs in the layer, d1 is the layer depth of current
		d1 := numFMs[layer]
		for i1 := 0; i1 < d1; i1++ { // this layer loop
			// The network is fully connected.  d2 is the layer depth of previous
			d2 := numFMs[layer-1]
			// filter (convolve) using the kernel in i1
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				// Convolve the previous layer y with the filter connecting
				// this layer to the previous layer.
				err := cnn.filterF(&cnn.link[layer-1][i2*d1+i1], layer-1, i1, i2)
				if err != nil {
					fmt.Printf("filter forward propagate error: %v\n", err.Error())
					return fmt.Errorf("filter forward propagate error: %v", err)
				}
			}
		}
	}

	// Flatten the last FM layer and insert into linear array, 32*16*16=8192
	// Take the downsampled FM from the temp FM
	// node[0][0] is the bias = 1, so skip k = 0
	cnn.node[0][0].y = 1.0
	k := 1
	n := len(numFMs) - 1
	for i := 0; i < numFMs[n]; i++ {
		for row := 0; row < sizeFMs[n]; row++ {
			for col := 0; col < sizeFMs[n]; col++ {
				cnn.node[0][k].y = cnn.fm[n][i].data[row][col].y
				k++
			}
		}
	}

	// Last layers uses flattened fully-connected MLP arrangement.
	// Propagate forward as in MLP
	d1 := len(cnn.node[1])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		// Each FM in previous layer is connected to current FM because
		// the network is fully connected.  d2 is the layer depth of previous
		d2 := len(cnn.node[0])
		// Loop over weights to get v
		v := 0.0
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			v += cnn.wgtOutput[i2*d1+i1] * cnn.node[0][i2].y
		}
		// compute output y = Phi(v) is the logistic function
		cnn.node[1][i1].y = 1.0 / (1.0 + math.Exp(-v))
	}

	return nil
}

func (cnn *CNN) propagateBackward() error {

	// output layer is different, no bias node, so the indexing is different
	// Loop over nodes in output layer
	d1 := len(cnn.node[1])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		//compute error e=d-Phi(v)
		cnn.node[1][i1].delta = cnn.desired[i1] - cnn.node[1][i1].y
		// Multiply error by this node's Phi'(v) to get local gradient.
		cnn.node[1][i1].delta *= cnn.node[1][i1].y * (1.0 - cnn.node[1][i1].y)
		// Send this node's local gradient to previous layer nodes through corresponding link.
		// Each node in previous layer is connected to current node because the network
		// is fully connected.  d2 is the previous layer depth
		d2 := len(cnn.node[0])
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			cnn.node[0][i2].delta += cnn.wgtOutput[i2*d1+i1] * cnn.node[1][i1].delta
			// Update weight with y and local gradient
			cnn.wgtOutput[i2*d1+i1] +=
				cnn.learningRate * cnn.node[1][i1].delta * cnn.node[0][i2].y

		}
		// Reset this local gradient to zero for next training example
		cnn.node[1][i1].delta = 0.0
	}

	// Insert the flattened output layer local gradients into the last Feature Map
	// Go from linear to planar data.  Upsample the local gradients 2x by inserting
	// avg in elements.  Skip k = 0 because that is the bias delta for the flattened layers.
	k := 1
	n := len(numFMs) - 1
	for i := 0; i < numFMs[n]; i++ {
		for row := 0; row < sizeFMs[n]; row++ {
			for col := 0; col < sizeFMs[n]; col++ {
				cnn.fm[n][i].data[row][col].delta = cnn.node[0][k].delta
				// Reset this local gradient to zero for next training example
				cnn.node[0][k].delta = 0.0
				k++
			}
		}
	}

	// Loop over layers in backward direction, starting at the last hidden layer
	for layer := n; layer > 0; layer-- {
		// Loop over FMs in this layer, d1 is the current layer depth
		d1 := len(cnn.fm[layer])
		for i1 := 0; i1 < d1; i1++ { // this layer loop
			// Multiply deltas propagated from downstream FMs by this node's Phi'(v) to get local gradient.
			// For the ReLU = max(0, v), Phi'(v) = 1 if v > 0, else Phi'(v) = 0
			for j := range cnn.fm[layer][i1].data {
				for i := range cnn.fm[layer][i1].data[j] {
					if cnn.fm[layer][i1].data[j][i].y <= 0 {
						cnn.fm[layer][i1].data[j][i].delta = 0.0
					}
				}
			}

			// Filter (convolve) this layer's delta and send to previous layers.
			// Each FM in previous layer is connected to current FM because the network
			// is fully connected.  d2 is the previous layer depth
			d2 := len(cnn.fm[layer-1])
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				// convolve the delta with the filter
				err := cnn.filterB(&cnn.link[layer-1][i2*d1+i1], layer, i1)
				if err != nil {
					fmt.Printf("filter backward propagate error: %v\n", err.Error())
					return fmt.Errorf("filter backward propagate error: %v", err)
				}

				// Update filter by convolving y from previous layer and upsampled local gradient
				err = cnn.updateFilter(layer-1, i1, i2, d1)
				if err != nil {
					fmt.Printf("updateFilter backward propagate error: %v\n", err.Error())
					return fmt.Errorf("updateFilter backward propagate error: %v", err)
				}
			}

			// Reset this local gradient to zero for next training example
			for i := range cnn.fm[layer][i1].data {
				for j := range cnn.fm[layer][i1].data[i] {
					cnn.fm[layer][i1].data[i][j].delta = 0.0
				}
			}
		}
	}
	return nil
}

// runEpochs performs forward and backward propagation over each sample
func (cnn *CNN) runEpochs() error {

	// Initialize the Filter 3x3 weights and flattened output weights

	// hidden layer filters, excluding the output layer
	// initialize the Filter wgt randomly, zero mean, normalize by fan-in
	for layer := range cnn.link {
		for n := range cnn.link[layer] {
			cnn.link[layer][n].biaswgt = 2.0 * (rand.Float64() - .5) / (kernelDim * kernelDim)
			for i := range cnn.link[layer][n].wgt {
				for j := range cnn.link[layer][n].wgt[i] {
					cnn.link[layer][n].wgt[i][j] = 2.0 * (rand.Float64() - .5) / (kernelDim * kernelDim)
					//cnn.link[layer][n].wgt[i][j] = rand.Float64() / (kernelDim * kernelDim)
				}
			}
		}
	}

	// output layer links
	for i := range cnn.wgtOutput {
		cnn.wgtOutput[i] = 2.0 * (rand.Float64() - .5) / (kernelDim * kernelDim)
		//cnn.wgtOutput[i] = rand.Float64() / (kernelDim * kernelDim)
	}

	for n := 0; n < cnn.epochs; n++ {
		// Loop over the training examples
		for _, samp := range cnn.samples {
			// Forward Propagation
			err := cnn.propagateForward(samp, n)
			if err != nil {
				return fmt.Errorf("forward propagation error: %s", err.Error())
			}

			// Backward Propagation
			err = cnn.propagateBackward()
			if err != nil {
				return fmt.Errorf("backward propagation error: %s", err.Error())
			}
		}

		// At the end of each epoch, loop over the output nodes and calculate mse
		cnn.calculateMSE(n)

		// Shuffle training exmaples
		rand.Shuffle(len(cnn.samples), func(i, j int) {
			cnn.samples[i], cnn.samples[j] = cnn.samples[j], cnn.samples[i]
		})
	}

	return nil
}

// init parses the html template files
func init() {
	tmplTrainingCNN = template.Must(template.ParseFiles(fileTrainingCNN))
	tmplTestingCNN = template.Must(template.ParseFiles(fileTestingCNN))
}

// createExamples creates a slice of training or testing examples
func (cnn *CNN) createExamples() error {
	// read in image files and convert to grayscale using color.GrayModel
	files, err := os.ReadDir(dataDir)
	if err != nil {
		fmt.Printf("ReadDir for %s error: %v\n", dataDir, err)
		return fmt.Errorf("ReadDir for %s error %v", dataDir, err.Error())
	}
	// Each image is a separate image class
	class := 0
	// display convention chosen: if gray.Y < on, set value to 1 means black
	// else if gray.Y >= on, set value to -1 means white
	for _, dirEntry := range files {
		name := dirEntry.Name()
		if filepath.Ext(dirEntry.Name()) == ".png" {
			f, err := os.Open(path.Join(dataDir, name))
			if err != nil {
				fmt.Printf("Open %s error: %v\n", name, err)
				return fmt.Errorf("file Open %s error: %v", name, err.Error())
			}
			defer f.Close()
			// only process classes files
			if class == classes {
				return fmt.Errorf("can only process %v png files", classes)
			}
			// convert PNG to image.Image
			img, err := png.Decode(f)
			if err != nil {
				fmt.Printf("Decode %s error: %v\n", name, err)
				return fmt.Errorf("image Decode %s error: %v", name, err.Error())
			}
			rect := img.Bounds()
			// save the name of the image without the ext
			cnn.samples[class].name = strings.Split(name, ".")[0]
			// The desired output of the CNN is class
			cnn.samples[class].desired = class
			for y := rect.Min.Y; y < rect.Max.Y; y++ {
				for x := rect.Min.X; x < rect.Max.X; x++ {
					gray := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
					cnn.samples[class].image[y][x] = gray.Y
				}
			}
			class++
		}
	}
	fmt.Printf("Read %d png files\n", class)

	return nil
}

// newCNN constructs an CNN instance for training
func newCNN(r *http.Request, epochs int, plot *PlotT) (*CNN, error) {
	// Read the training parameters in the HTML Form

	txt := r.FormValue("learningrate")
	learningRate, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("learningrate float conversion error: %v\n", err)
		return nil, fmt.Errorf("learningrate float conversion error: %s", err.Error())
	}

	cnn := CNN{
		epochs:       epochs,
		learningRate: learningRate,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		samples: make([]Sample, classes),
	}

	// construct container for images
	for i := range cnn.samples {
		cnn.samples[i].image = make([][]uint8, imgHeight)
		for j := range cnn.samples[i].image {
			cnn.samples[i].image[j] = make([]uint8, imgWidth)
		}
	}

	// *********** Links ******************************************************

	// construct links that hold the filters
	cnn.link = make([][]Filter, hiddenLayers)
	// hidden layer filters, excluding the output layer
	// previous layer FM depth
	m := 1
	for i, n := range numFMs[1:] {
		// fully-connected hidden layers
		k := m * n
		cnn.link[i] = make([]Filter, k)
		for j := 0; j < k; j++ {
			cnn.link[i][j] = Filter{n: j, layer: i}
		}
		m = n
	}

	// outer layer uses fully connected MLP using flattened last
	// hidden layer Feature Maps = 32*(16)*(16) = 8192, using downsampled FM
	// added bias weight with constant input = 1
	i := len(numFMs) - 1
	m = numFMs[i]*sizeFMs[i]*sizeFMs[i] + 1
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// output layer links
	cnn.wgtOutput = make([]float64, olnodes*m)

	// ******************* Feature Maps and Nodes ****************************
	// Input and hidden layer Feature Maps, flattened output layer nodes
	cnn.fm = make([][]FeatureMap, len(numFMs))
	cnn.node = make([][]Node, 2)

	//  input and hidden layers
	for i := 0; i < len(numFMs); i++ {
		cnn.fm[i] = make([]FeatureMap, numFMs[i])
		for j := 0; j < numFMs[i]; j++ {
			nodes := make([][]Node, sizeFMs[i])
			for k := 0; k < sizeFMs[i]; k++ {
				nodes[k] = make([]Node, sizeFMs[i])
			}
			cnn.fm[i][j] = FeatureMap{h: sizeFMs[i], w: sizeFMs[i], data: nodes}
		}
	}

	// next to last layer is the flattended last hidden layer
	cnn.node[0] = make([]Node, m)
	// init bias node to 1
	cnn.node[0][0].y = 1.0

	// output layer, which has no bias node
	cnn.node[1] = make([]Node, olnodes)

	// *************************************************************

	// construct desired from classes, binary representation
	cnn.desired = make([]float64, olnodes)

	// mean-square error
	cnn.mse = make([]float64, epochs)

	return &cnn, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (cnn *CNN) gridFillInterp() error {
	var (
		x            float64 = 0.0
		y            float64 = cnn.mse[0]
		prevX, prevY float64
		xscale       float64
		yscale       float64
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (cnn.xmax - cnn.xmin)
	yscale = float64(rows-1) / (cnn.ymax - cnn.ymin)

	cnn.plot.Grid = make([]string, rows*cols)

	// This cell location (row,col) is on the line
	row := int((cnn.ymax-y)*yscale + .5)
	col := int((x-cnn.xmin)*xscale + .5)
	cnn.plot.Grid[row*cols+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := cnn.ymax - cnn.ymin
	lenEPx := cnn.xmax - cnn.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < cnn.epochs; i++ {
		x++
		// ensemble average of the mse
		y = cnn.mse[i]

		// This cell location (row,col) is on the line
		row := int((cnn.ymax-y)*yscale + .5)
		col := int((x-cnn.xmin)*xscale + .5)
		cnn.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(float64(cols) * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(float64(rows) * lenEdgeY / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := ncellsX
		if ncellsY > ncells {
			ncells = ncellsY
		}

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((cnn.ymax-interpY)*yscale + .5)
			col := int((interpX-cnn.xmin)*xscale + .5)
			cnn.plot.Grid[row*cols+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (cnn *CNN) insertLabels() {
	cnn.plot.Xlabel = make([]string, xlabels)
	cnn.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (cnn.xmax - cnn.xmin) / (xlabels - 1)
	x := cnn.xmin
	// First label is empty for alignment purposes
	for i := range cnn.plot.Xlabel {
		cnn.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (cnn.ymax - cnn.ymin) / (ylabels - 1)
	y := cnn.ymin
	for i := range cnn.plot.Ylabel {
		cnn.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// handleTraining performs forward and backward propagation to calculate the weights
func handleTrainingCNN(w http.ResponseWriter, r *http.Request) {

	var (
		plot PlotT
		cnn  *CNN
	)

	// Get the number of epochs
	txt := r.FormValue("epochs")
	// Need epochs to continue
	if len(txt) > 0 {
		epochs, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Epochs int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Epochs conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create CNN instance to hold state
		cnn, err = newCNN(r, epochs, &plot)
		if err != nil {
			fmt.Printf("newCNN() error: %v\n", err)
			plot.Status = fmt.Sprintf("newCNN() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Create training examples by reading in the encoded characters
		err = cnn.createExamples()
		if err != nil {
			fmt.Printf("createExamples error: %v\n", err)
			plot.Status = fmt.Sprintf("createExamples error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Loop over the Epochs
		err = cnn.runEpochs()
		if err != nil {
			fmt.Printf("runEnsembles() error: %v\n", err)
			plot.Status = fmt.Sprintf("runEnsembles() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put MSE vs Epoch in PlotT
		err = cnn.gridFillInterp()
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		cnn.insertLabels()

		// At the end of all epochs, insert form previous control items in PlotT
		cnn.plot.LearningRate = strconv.FormatFloat(cnn.learningRate, 'f', 5, 64)
		cnn.plot.Epochs = strconv.Itoa(cnn.epochs)

		// Save Filters to csv file, one  per line
		f, err := os.Create(path.Join(dataDir, fileweights))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(fileweights), err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", path.Join(fileweights), err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer f.Close()

		// Save epochs and learning rate
		fmt.Fprintf(f, "%d,%f\n", cnn.epochs, cnn.learningRate)

		// Save the weights in cnn.link
		// Save Filters, 3x3 kernel and bias weights, one filter per line
		// i is the hidden layer, n is the number of Feature Maps in this layer
		for i, n := range numFMs[1:] {
			// loop over the Feature Maps in this layer
			for j := 0; j < n; j++ {
				for row := 0; row < kernelDim; row++ {
					for col := 0; col < kernelDim; col++ {
						fmt.Fprintf(f, "%.10f,", cnn.link[i][j].wgt[row][col])
					}
				}
				// last one is the bias weight and newline
				fmt.Fprintf(f, "%.10f\n", cnn.link[i][j].biaswgt)
			}
		}

		// save flattened layer, one weight per line because too long to split
		for _, wt := range cnn.wgtOutput {
			fmt.Fprintf(f, "%.10f\n", wt)
		}

		cnn.plot.Status = "MSE plotted"

		// Execute data on HTML template
		if err = tmplTrainingCNN.Execute(w, cnn.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Epochs and Learning Rate parameters."
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	}
}

// Classify test examples and display test results
func (cnn *CNN) runClassification() error {
	// Loop over the training examples
	cnn.plot.Grid = make([]string, rows*cols)
	cnn.statistics =
		Stats{correct: make([]int, classes), classCount: make([]int, classes)}
	for i, samp := range cnn.samples {
		// Forward Propagation
		err := cnn.propagateForward(samp, 1)
		if err != nil {
			return fmt.Errorf("forward propagation error: %s", err.Error())
		}
		// At output layer, classify example, increment class count, %correct
		// Convert node output y to class
		err = cnn.determineClass(i, samp)
		if err != nil {
			return fmt.Errorf("determineClass error: %s", err.Error())
		}
	}

	cnn.plot.TestResults = make([]Results, classes)

	totalCount := 0
	totalCorrect := 0
	// tabulate TestResults by converting numbers to string in Results
	for i := range cnn.plot.TestResults {
		totalCount += cnn.statistics.classCount[i]
		totalCorrect += cnn.statistics.correct[i]
		cnn.plot.TestResults[i] = Results{
			Class:   strconv.Itoa(i),
			Image:   cnn.samples[i].name,
			Count:   strconv.Itoa(cnn.statistics.classCount[i]),
			Correct: strconv.Itoa(cnn.statistics.correct[i] * 100 / cnn.statistics.classCount[i]),
		}
	}
	cnn.plot.TotalCount = strconv.Itoa(totalCount)
	cnn.plot.TotalCorrect = strconv.Itoa(totalCorrect * 100 / totalCount)
	cnn.plot.LearningRate = strconv.FormatFloat(cnn.learningRate, 'f', 5, 64)
	cnn.plot.Epochs = strconv.Itoa(cnn.epochs)

	cnn.plot.Status = "Testing results completed."

	return nil
}

// drawImages draws the images that are classified
func (cnn *CNN) drawImages(r *http.Request) error {

	// positioning values on the canvas
	const (
		startRow = (rows - imgHeight) / 2
		startCol = (cols - imgWidth) / 2
		on       = 128 // boundary between black and white for 8-bit grayscale
	)

	txt := r.FormValue("selectimage")
	samp := 0
	var err error = nil
	if len(txt) > 0 {
		samp, err = strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("image int conversion error: %v\n", err)
			return fmt.Errorf("image int conversion error: %s", err.Error())
		}
	}

	// insert black and white image in TestResults
	current := startRow*cols + startCol
	// draw the flattened image
	k := 0
	for j := 0; j < imgHeight; j++ {
		for i := 0; i < imgWidth; i++ {
			// This cell is black in the image
			if cnn.samples[samp].image[j][i] < on {
				cnn.plot.Grid[current+i] = "online"
			}
			k++
		}
		current += cols
	}

	return nil
}

// newTestingCNN constructs an CNN from the saved cnn weights and parameters
func newTestingCNN(r *http.Request, plot *PlotT) (*CNN, error) {
	// Read in weights from csv file, ordered by layers and Feature Maps
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	// construct the CNN
	cnn := CNN{
		plot:    plot,
		samples: make([]Sample, classes),
	}
	// construct container for images
	for i := range cnn.samples {
		cnn.samples[i].image = make([][]uint8, imgHeight)
		for j := range cnn.samples[i].image {
			cnn.samples[i].image[j] = make([]uint8, imgWidth)
		}
	}

	// *********** Links ******************************************************

	// construct links that hold the filters
	cnn.link = make([][]Filter, hiddenLayers)
	// hidden layer filters, excluding the output layer
	// previous layer FM depth
	m := 1
	for i, n := range numFMs[1:] {
		// fully-connected hidden layers
		k := m * n
		cnn.link[i] = make([]Filter, k)
		for j := 0; j < k; j++ {
			cnn.link[i][j] = Filter{n: j, layer: i}
		}
		m = n
	}

	// outer layer uses fully connected MLP using flattened last
	// hidden layer Feature Maps = 32*16*16 = 8192, using downsampled FM
	// add bias weight with constant input = 1
	i := len(numFMs) - 1
	m = numFMs[i]*sizeFMs[i]*sizeFMs[i] + 1
	olnodes := int(math.Ceil(math.Log2(float64(classes))))
	N := m * olnodes

	// output layer links
	cnn.wgtOutput = make([]float64, N)

	// ******************* Feature Maps and Nodes ****************************
	// Input and hidden layer Feature Maps, flattened output layer nodes
	cnn.fm = make([][]FeatureMap, len(numFMs))
	cnn.node = make([][]Node, 2)

	//  input and hidden layers
	for i := 0; i < len(numFMs); i++ {
		cnn.fm[i] = make([]FeatureMap, numFMs[i])
		for j := 0; j < numFMs[i]; j++ {
			nodes := make([][]Node, sizeFMs[i])
			for k := 0; k < sizeFMs[i]; k++ {
				nodes[k] = make([]Node, sizeFMs[i])
			}
			cnn.fm[i][j] = FeatureMap{h: sizeFMs[i], w: sizeFMs[i], data: nodes}
		}
	}

	// next to last layer is the flattended last hidden layer
	cnn.node[0] = make([]Node, m)
	// init bias node to 1
	cnn.node[0][0].y = 1.0

	// output layer, which has no bias node
	cnn.node[1] = make([]Node, olnodes)

	// *************************************************************

	scanner := bufio.NewScanner(f)

	// Read in epochs and learning rate
	scanner.Scan()
	line := scanner.Text()
	items := strings.Split(line, ",")

	epochs, err := strconv.Atoi(items[0])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v", items[0], err)
		return nil, err
	}
	cnn.epochs = epochs

	learningRate, err := strconv.ParseFloat(items[1], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v", items[3], err)
		return nil, err
	}
	cnn.learningRate = learningRate

	// retrieve the weights and insert in cnn.link
	// Read Filters, 3x3 kernel and bias weights
	// i is the hidden layer, n is the number of Feature Maps in this layer
loop:
	for i, n := range numFMs[1:] {
		// loop over the Feature Maps in this layer
		for j := 0; j < n; j++ {
			ok := scanner.Scan()
			if !ok {
				break loop
			}
			line := scanner.Text()
			weights := strings.Split(line, ",")
			cnn.link[i][j] = Filter{n: j, layer: i}
			k := 0
			for row := 0; row < kernelDim; row++ {
				for col := 0; col < kernelDim; col++ {
					wt, err := strconv.ParseFloat(weights[k], 64)
					if err != nil {
						fmt.Printf("ParseFloat of %s error: %v", weights[k], err)
						k++
						continue
					}
					cnn.link[i][j].wgt[row][col] = wt
					k++
				}
			}
			// last one is the bias weight
			wt, err := strconv.ParseFloat(weights[k], 64)
			if err != nil {
				fmt.Printf("ParseFloat of %s error: %v", weights[k], err)
				k++
				continue
			}
			cnn.link[i][j].biaswgt = wt
		}
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	// last layer, one weight per line
	for i := 0; i < N; i++ {
		ok := scanner.Scan()
		if !ok {
			break
		}
		line := scanner.Text()
		wgt, err := strconv.ParseFloat(line, 64)
		if err != nil {
			fmt.Printf("ParseFloat error: %v\n", err.Error())
			continue
		}
		cnn.wgtOutput[i] = wgt
	}

	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	// *******************************************************

	// construct desired from classes, binary representation
	cnn.desired = make([]float64, olnodes)

	return &cnn, nil
}

// handleTesting performs pattern classification of the test data
func handleTestingCNN(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		cnn  *CNN
		err  error
	)
	// Construct CNN instance containing CNN state
	cnn, err = newTestingCNN(r, &plot)
	if err != nil {
		fmt.Printf("newTestingCNN() error: %v\n", err)
		plot.Status = fmt.Sprintf("newTestingCNN() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Create testing examples by reading in the images
	err = cnn.createExamples()
	if err != nil {
		fmt.Printf("createExamples error: %v\n", err)
		plot.Status = fmt.Sprintf("createExamples error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// At end of all examples tabulate TestingResults
	// Convert numbers to string in Results
	err = cnn.runClassification()
	if err != nil {
		fmt.Printf("runClassification() error: %v\n", err)
		plot.Status = fmt.Sprintf("runClassification() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Draw the images to show what is being classified
	err = cnn.drawImages(r)
	if err != nil {
		fmt.Printf("drawImages() error: %v\n", err)
		plot.Status = fmt.Sprintf("drawImages() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Execute data on HTML template
	if err = tmplTestingCNN.Execute(w, cnn.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for training and testing the CNN Neural Network

	// Create HTTP handler for training
	http.HandleFunc(patternTrainingCNN, handleTrainingCNN)
	// Create HTTP handler for testing
	http.HandleFunc(patternTestingCNN, handleTestingCNN)
	fmt.Printf("Convolutional Neural Network Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
