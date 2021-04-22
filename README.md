# mnist-torch
Solving MNIST with Pytorch (Machine Vision Hello World)

How do we teach a computer to read handwriting?

Based on this blog post:
https://nextjournal.com/gkoehler/pytorch-mnist

This is one of the original machine vision problems so it's a good one to start with.

To train our AI we need examples of handwritten digits. MNIST is the most common dataset.

We can then feed each handwritten number image into the neural network. This one's called a convolutional neural network. All it sees is each photograph of a number as it's pixel values.

The network contains multiple layers of mathematical magic that slowly teach the network hidden similarities between each handwritten number.

It then generates a prediction based on its assumptions as to what that digit is and then checks to see if it was right

We repeated this process 6 times and we get an accuracy of 97%.

We then tested it on 6 sample images from the dataset with 100% accuracy.
