from __future__ import print_function
####import all libraries needeed ##########
import numpy as np

#some recommended libraries for image handling (you should only need one of these)

from skimage import data
from PIL import Image
import time
import matplotlib.pyplot as plt
import numpy as np

from conv import Conv2D

def taskB():
    im = Image.open('checkerboard.png')
    im = np.array(im)


    x = np.arange(1, 10)
    y = []

    for i in range(1, 10):

        conv2D = Conv2D(3, i, 3, 1, 'rand', 8)
        start = time.time()
        conv2D.forward(im)
        end = time.time()
        y.append(end-start)
        print("time for " + str(i) + " channels: " + str(end - start))

    plt.xlabel('Kernel Size')
    plt.ylabel('Number of operations')
    plt.title('Task B')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig('taskB_chart2.png')

def taskC():

    im = Image.open('checkerboard.png')
    im = np.array(im)


    x = [3, 5, 7, 11, 13, 15]
    y = []

    for i in x:

        conv2D = Conv2D(3, 3, i, 1, 'rand', 9)
        print("Kernel Size: " + str(i))

        [num_ops, _] = conv2D.forward(im)
        y.append(num_ops)

        print("num ops for kernel size of  " + str(i) + ": " + str(num_ops))

    plt.xlabel('Output Channel Size')
    plt.ylabel('Elapsed Time (s)')
    plt.title('Task B')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig('taskC_chart2.png')

if __name__ == "__main__":

    #im = Image.open('flower.png')
    #im = np.array(im)

    taskC()

#Conv2D(in_channel, out_channel, kernel_size, stride, mode, task)


#### Initialize object of class Conv2D ########

    #conv2d = Conv2D(in_channel=1, o_channel=1, kernel_size=3, stride=1, mode="known", task = 1)

#### Call forward method of Conv2D object #####

    #[num_ops, out_image] = conv2d.forward(im)

##### Show/save output image of forward method #####

    #print(num_ops)
    #o = Image.fromarray(out_image)
    #o.save('temp.png')






