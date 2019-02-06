####import all libraries needeed ##########

import numpy as np
from PIL import Image

import time

#### Convolution Class ########

# Defined Kernels
identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
K1 = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
K2 = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
K3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
K4 = [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
K5 = [[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]]
K6 = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
K7 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
K8 = [[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]]


class Conv2D:

    def __init__(self, in_channel, o_channel, kernel_size, stride, mode, task):
        ### initialization function of Convolution Class (no output)

        self.in_channel = in_channel
        self.out_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.task = task
        self.kernels = []

        # Initialize Kernels for each task
        if mode == 'known' :
            if task == 0:
                self.kernels = [identity]
            if task == 1:
                self.kernels = [K1]
            if task == 2:
                self.kernels = [K2]
            if task == 3:
                self.kernels = [K3]
            if task == 4:
                self.kernels = [K4]
            if task == 5:
                self.kernels = [K5]
            if task == 6:
                self.kernels = [K6]
            if task == 7:
                self.kernels = [K7, K8]


        else:
            if task == 8:

                self.kernels = []
                for i in range(0, self.out_channel):
                    r = np.random.randint(low=-100, high=100, size=(kernel_size, kernel_size))
                    self.kernels.append(r)

                print("Num kernerls:" + str(len(self.kernels)))

            if task == 9:
                k1 = np.random.randint(low=-100, high=100, size=(kernel_size, kernel_size))
                k2 = np.random.randint(low=-100, high=100, size=(kernel_size, kernel_size))
                k3 = np.random.randint(low=-100, high=100, size=(kernel_size, kernel_size))

                self.kernels = [k1, k2, k3]

    def conv_Greyscale_1channel(self, im, kernel):

        print("conv greyscale 1 channel")
        # Convert image to greyscale
        im = im.convert('L')

        # initialize num ops to 0
        num_ops = 0

        # Get the size of the image
        width, height = im.size

        width = int(width / self.stride)
        height = int(height / self.stride)


        # Create a output image numpy array
        out_im = np.zeros([width, height, 3], dtype=np.uint8)

        # Set the offset to half the kernel size floored
        offset = int(self.kernel_size / 2)

        row = 0
        col = 0

        while((row + self.kernel_size) < height):

            while((col + self.kernel_size) < width):

                # Define a bounding box
                box = (row, col, row + self.kernel_size, col + self.kernel_size)

                # RBG bands should be the same
                band = im.crop(box)

                # Perform convolution - 9 multiplications, 9 additions
                o_channel = np.sum(np.multiply(band, kernel))

                # Increment num_ops
                num_ops = num_ops + 2 * self.kernel_size * self.kernel_size

                # Grey scale will have the same value, depth 3 for pillow conversion later
                out_im[col + offset][row + offset][0] = o_channel
                out_im[col + offset][row + offset][1] = o_channel
                out_im[col + offset][row + offset][2] = o_channel

                # Move the kernel right by the stride value
                col = col + self.stride

            col = 0
            row = row + 1

        return [num_ops, out_im]

    def conv_RGB_1channel(self, im, kernel):

        im = im.convert('RGB')

        print("conv RGB 1 channel")
        r, g, b = (0, 1, 2)

        num_ops = 0
        # Get the image dimensions
        height, width = im.size

        # Create a output image numpy array
        out_im = np.zeros([width, height, 3], dtype=np.uint8)

        offset = int(self.kernel_size/2)

        # Initialize Loop control variables
        row = 0
        col = 0

        while((row + self.kernel_size) < height):

            while((col + self.kernel_size) < width):

                # Define a bounding box
                box = (row, col, row + self.kernel_size, col + self.kernel_size)

                # Split bands / channels
                r_band, g_band, b_band = im.crop(box).split()

                # Turn PIL images into numpy arrays
                r_array = np.array(r_band)
                g_array = np.array(g_band)
                b_array = np.array(b_band)

                # Perform Convolution
                r_channel = np.sum(np.multiply(r_array, kernel))
                g_channel = np.sum(np.multiply(g_array, kernel))
                b_channel = np.sum(np.multiply(b_array, kernel))

                # 9 additions, 9 multiplications, 3 times
                num_ops = num_ops + self.kernel_size * self.kernel_size * 3

                # Save the output image
                #print(row + offset, col + offset, (r_channel, g_channel, b_channel))

                sum_ = r_channel + g_channel + b_channel
                out_im[col + offset][row + offset][r] = sum_
                out_im[col + offset][row + offset][g] = sum_
                out_im[col + offset][row + offset][b] = sum_

                col = col + self.stride

            col = 0
            row += 1

        return [num_ops, out_im]

    def conv_RBG_2channel(self, im, kernels):

        im = im.convert('RGB')

        print("Conv RGB 2 channel")

        num_ops = 0

        r, g, b = (0, 1, 2)

        # Get the image dimensions
        height, width = im.size

        # Create a output image numpy array
        out_im1 = np.zeros([width, height, 3], dtype=np.uint8)
        out_im2 = np.zeros([width, height, 3], dtype=np.uint8)

        offset = int(self.kernel_size/2)

        # Initialize Loop control variables
        row = 0
        col = 0

        while((row + self.kernel_size) < height):

            while((col + self.kernel_size) < width):

                # Define a bounding box
                box = (row, col, row + self.kernel_size, col + self.kernel_size)

                # Split bands / channels
                r_band, g_band, b_band = im.crop(box).split()

                # Turn PIL images into numpy arrays
                r_array = np.array(r_band)
                g_array = np.array(g_band)
                b_array = np.array(b_band)

                # Perform the first convolution
                r_channel1 = np.sum(np.multiply(r_array, kernels[0]))
                g_channel1 = np.sum(np.multiply(g_array, kernels[0]))
                b_channel1 = np.sum(np.multiply(b_array, kernels[0]))


                # 9 multiplications, 9 additions
                num_ops = num_ops + 9 + 9

                # Perform the second convolution
                r_channel2 = np.sum(np.multiply(r_array, kernels[1]))
                g_channel2 = np.sum(np.multiply(g_array, kernels[1]))
                b_channel2 = np.sum(np.multiply(b_array, kernels[1]))
                num_ops = num_ops + 9 + 9

                # Calculate the sums
                sum1 = r_channel1 + g_channel1 + b_channel1
                sum2 = r_channel2 + g_channel2 + b_channel2

                # 6 more additions
                num_ops = num_ops + self.kernel_size + self.kernel_size


                out_im1[col + offset][row + offset][r] = sum1
                out_im1[col + offset][row + offset][g] = sum1
                out_im1[col + offset][row + offset][b] = sum1

                out_im2[col + offset][row + offset][r] = sum2
                out_im2[col + offset][row + offset][g] = sum2
                out_im2[col + offset][row + offset][b] = sum2

                col = col + self.stride

            col = 0
            row += 1

        return [num_ops,[out_im1, out_im2]]

    def increasing_output_channels(self, im):

        im = im.convert('RGB')
        print("increasing output channels")

        num_ops = 0

        r, g, b = (0, 1, 2)

        # Get the image dimensions
        height, width = im.size

        # Create a output image numpy array
        out_im1 = np.zeros([width, height, 3], dtype=np.uint8)
        out_im2 = np.zeros([width, height, 3], dtype=np.uint8)

        offset = int(self.kernel_size/2)

        # Initialize Loop control variables
        row = 0
        col = 0

        while((row + self.kernel_size) < height):

            while((col + self.kernel_size) < width):

                # Define a bounding box
                box = (row, col, row + self.kernel_size, col + self.kernel_size)

                # Split bands / channels
                r_band, g_band, b_band = im.crop(box).split()

                # Turn PIL images into numpy arrays
                r_array = np.array(r_band)
                g_array = np.array(g_band)
                b_array = np.array(b_band)

                # Repeat convolution for each output channel / random 3x3 kernel


                for i in range(0, self.out_channel):

                    # Perform the first convolution
                    #print(self.kernels[i])
                    r_channel1 = np.sum(np.multiply(r_array, self.kernels[i]))
                    g_channel1 = np.sum(np.multiply(g_array, self.kernels[i]))
                    b_channel1 = np.sum(np.multiply(b_array, self.kernels[i]))

                    # 9 multiplications, 9 additions
                    num_ops = num_ops + self.kernel_size * self.kernel_size

                col = col + self.stride

            col = 0
            row += 1

                # For Part B, do not need to show images
                # Just need to record times

        return num_ops

    def increasing_kernel_sizes(self, im):


        im = im.convert('RGB')
        print("increasing output channels")

        num_ops = 0

        r, g, b = (0, 1, 2)

        # Get the image dimensions
        height, width = im.size

        # Create a output image numpy array
        out_im1 = np.zeros([width, height, 3], dtype=np.uint8)
        out_im2 = np.zeros([width, height, 3], dtype=np.uint8)

        offset = int(self.kernel_size / 2)

        # Initialize Loop control variables
        row = 0
        col = 0

        while ((row + self.kernel_size) < height):

            while ((col + self.kernel_size) < width):
                # Define a bounding box
                box = (row, col, row + self.kernel_size, col + self.kernel_size)

                # Split bands / channels
                r_band, g_band, b_band = im.crop(box).split()

                # Turn PIL images into numpy arrays
                r_array = np.array(r_band)
                g_array = np.array(g_band)
                b_array = np.array(b_band)

                # Repeat convolution for each output channel / random 3x3 kernel
                # Perform the first convolution
                for k in self.kernels:
                    r_channel1 = np.sum(np.multiply(r_array, k))
                    g_channel1 = np.sum(np.multiply(g_array, k))
                    b_channel1 = np.sum(np.multiply(b_array, k))

                    num_ops = num_ops + (self.kernel_size * self.kernel_size) * 3

                col = col + 1

            col = 0
            row = row + 1

        return num_ops

    def forward(self, input_image):

        # Determine which function to call based on task
        input_image = Image.fromarray(input_image)

        im = None

        if self.task == 0 or self.task == 1 or self.task == 2 or self.task == 3 or self.task == 4 or self.task == 5:

            print(self.kernels)

            [ops, im] = self.conv_Greyscale_1channel(input_image, self.kernels[0])

        if self.task == 6 :

            print(self.kernels)

            [ops, im] = self.conv_RGB_1channel(input_image, self.kernels[0])

        if self.task == 7 :

            [ops, im] = self.conv_RBG_2channel(input_image, self.kernels)


        if self.task == 8:

            ops = self.increasing_output_channels(input_image)
            im = None

        if self.task == 9:

            ops = self.increasing_kernel_sizes(input_image)
            im = None

        number_of_operations = ops
        output_image = im


        return number_of_operations, output_image



