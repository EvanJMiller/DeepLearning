
Using the MNIST dataset I trained networks for both the neural network developed earlier and the NnImg2Num that uses
Torch's NN Module. Both networks used two hidden linear layers of sizes (874, 128) and (128, 10). The network from the 
last homework assignment did not use batches and ended up taking too long to iterate over epochs to conclude whether or
not it would converge. However when tracking the MSE it did tend to lower over time but the accuracy did not exceed 15%
after running for roughly an hour. 

Using Torch's neural net was a different story. Splitting the data into batches of size 64 for training allowed iteration
of the 60,000 images per set to be computed much faster. I used a softmax function on the output and after 100 epochs 
the network reached upwards of 90% on some runs. The accuracy over time can be seen in the 'nn_accuracy.png' image, and 
the loss over time can be viewed in the 'loss.png' image.
