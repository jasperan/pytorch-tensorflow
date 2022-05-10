# Working with Data in TensorFlow

Welcome to the third article of this article series! In this article we'll deep dive into setting up a simple project using TensorFlow, as opposed to what we've been doing until now in the article series (working with PyTorch); hopefully we'll see how relatively easy developing in both libraries is, compared to each other.

As a reminder, let's have some initial information about TensorFlow. TensorFlow was created in 2015 by Google, and it's an open-source platform. From this original platform, multiple libraries have been developed to allow the use of the TensorFlow platform in various programming languages, like Python, JavaScript and even mobile devices. It provides a comprehensive ecosystem of tools for developers and researchers who want to bring the newest technologies from Machine Learning into reality. 

## Setting up OCI

To set up TensorFlow in OCI, we'll create a Data Science notebook in OCI, and we'll be able to access that notebook from the Cloud, saving us the trouble of setting up locally to start working on our project. This process that we'll follow is very similar to the steps we followed [in the first article from the series](https://github.com/jasperan/pytorch-tensorflow/1_getting_started_with_pytorch_on_oci.md). Make sure to check that article out if you run into trouble setting up a Data Science notebook in OCI.

First of all, we set up an OCI Data Science environment and instantiate the Data Science, by heading over to the OCI console and navigating to __OCI Data Science__:

![1](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/1.PNG?raw=true)

We navigate to our projects. Inside the only project we've created so far, we can have several notebook sessions; and these sessions have their own respective storage each. In our project, we'll create a new notebook session and we'll install the TensorFlow environment inside it.

![3](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/3.PNG?raw=true)

Accessing the notebook, we install the TensorFlow environment from the official Environment Explorer:

![10](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/10.PNG?raw=true)

o install it, we run the command with the corresponding pre-built identifier in a terminal. At the time of writing this article, TensorFlow 2.7 is the latest version available for CPUs; and TensorFlow 2.6 for GPUs:

## TensorFlow Fundamentals - Data Loading with Keras

It's important to notice that, [in the first article from the series](https://github.com/jasperan/pytorch-tensorflow/1_getting_started_with_pytorch_on_oci.md) we loaded an initial dataset (the well-known iris dataset) and created a Neural Network from scratch, in order to make accurate predictions about the type of irises (setosa, versicolor or virginica). We did this through the PyTorch library, that included a custom-built dataset loader.

Just like in PyTorch, TensorFlow also "likes" having their datasets loaded into tensors; however, [the official documentation](https://www.tensorflow.org/tutorials) points towards using **keras** as the data-loading library. It's pretty common to find Keras and TensorFlow being used interchangeably, as Keras eases the use of TensorFlow.

You'll ask yourself: why change the mechanism in which we load data? Both Keras and tensorFlow are machine learning NN technologies / libraries. However, Keras __encapsulates__ TensorFlow. It's a wrapper of the TensorFlow library, and it aims to make the development of a NN much easier. However, this comes with a cost, which is speed. If you decide to use Keras instead of TensorFlow, consider that anything built on top of the standard TensorFlow library will cause inaccuracies in terms of the model's throughput / machine prediction throughput, since everything is being filtered through Keras before accessing TensorFlow itself.
Having this in mind, you're free to load data using any data loading library available.

So, with efficiency and throughput in mind, I chose to just use Keras as a data-loading library, and not the library we've used before for development (including the iris NN).


So, here's an example of how easy data loading is using the Keras library. For this example, we're going to use the MNIST digits classification dataset. This is a dataset of sixty thousand grayscale images with dimensions of 28x28 pixels, where each picture represents images of digits 0 through 9, and with a test set size of 10.000 images. the __load_data__ function automatically divides the mnist dataset into this train/test size:

```python
import tensorflow as tf
mnist_dataset = tf.keras.datasets.mnist

# we split into train and tests using the load_data function 
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```
```bash
>>> Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
>>> 11493376/11490434 [==============================] - 0s 0us/step
>>> 11501568/11490434 [==============================] - 0s 0us/step
```

A question may arise why we're dividing the data by 255.0. This is done to normalize each pixel value (which ranges in the dataset from 0 to 255) into a value from 0 to 1 (basically we're applying normalization). It's also fair to say that this isn't necessary, as the NN will eventually learn by itself to normalize the data anyways. However, in order to establish good software practices and to handle the input data ourselves, we apply this very basic normalization before processing data.

If we print the current data that we have, we get an array of 28x28 as input data (our features) and an integer number as label (the number pixels actually represent):

![normalized](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/normalized.PNG?raw=true)

## Training the Model

Now, we're going to focus on training the model. As we said, every row of data represents an image. The size of this image is 28x28 pixels, which equals 784. Therefore, we're going to create a TensorFlow model with 784 inputs, and connect these inputs to a dense layer (hidden layer), which means that we're creating a recurrent NN. Therefore, automatic differentiation and complex gradient calculation will be performed internally and automatically. The hidden layer will have 128 nodes and ReLU activation; and will then be connected to a 10 node output layer (another dense layer).

It's very complex and really up to the Data Scientist to choose the initial parameters. For instance, when I started learning about AI and ML I always wondered how people came up with initial parameters, which could be considered as a challenging task. Talking with people who work with Neural Networks very extensively, I was told that this initial number is often performed semi-randomly (with some ideas in mind), without giving it too much thought. If the NN is able to learn with these initial parameters, we don't need to perform anything else, and if the NN is having a hard time learning through the specified number of epochs, we can do some things (which can be considered performing hyperparametrization in Machine Learning):
- Change the input parameters, like the number of nodes that compose a hidden layer, and try with different values
- Change the architecture of the deep NN (reminder: a deep NN is just a NN with more than one hidden layer)
- Change the loss function
- Change the optimizer
- Change the activation function function

Performing these things optimally takes a lot of practice and trial/error, so don't be discouraged to start, even if you don't know which values to input at the beginning.

```python
model = tf.keras.models.Sequential([
        # 1st layer: flatten the input. From a 28x28 array, we get a vector of 784 elements
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # 2nd layer: we create a dense layer with 128 nodes and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        # 3rd layer: we regularize the model (more about this below*)
        tf.keras.layers.Dropout(0.2),
        # 4th layer: we finish with another dense layer with 10 nodes.
        tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=1024)
```
This trains our model for 10 epochs. If loss didn't decrease so significantly, we could try increasing the number of epochs. Remember that, for each epoch, we'll have the NN's parameters updated by the previous iteration and updated, in order to give feedback to the NN and improve over time. I've personally seen models that ran for hundreds of thousands of epochs over a span of weeks, so don't be scared to add epochs. In theory, the more epochs you include, the better the accuracy will be for your dataset (generally speaking).
```bash
>>> Epoch 1/10
59/59 [==============================] - 1s 5ms/step - loss: 2.9121 - accuracy: 0.3993
>>> Epoch 2/10
59/59 [==============================] - 0s 6ms/step - loss: 1.1435 - accuracy: 0.8094
>>> Epoch 3/10
59/59 [==============================] - 0s 5ms/step - loss: 0.7380 - accuracy: 0.8539
>>> Epoch 4/10
59/59 [==============================] - 0s 6ms/step - loss: 0.5890 - accuracy: 0.8713
>>> Epoch 5/10
59/59 [==============================] - 0s 5ms/step - loss: 0.5113 - accuracy: 0.8818
>>> Epoch 6/10
59/59 [==============================] - 0s 5ms/step - loss: 0.4630 - accuracy: 0.8894
>>> Epoch 7/10
59/59 [==============================] - 0s 5ms/step - loss: 0.4298 - accuracy: 0.8949
>>> Epoch 8/10
59/59 [==============================] - 0s 5ms/step - loss: 0.4052 - accuracy: 0.8989
>>> Epoch 9/10
59/59 [==============================] - 0s 5ms/step - loss: 0.3864 - accuracy: 0.9025
>>> Epoch 10/10
59/59 [==============================] - 0s 5ms/step - loss: 0.3712 - accuracy: 0.9053

```

*The dropout layer introduced above is a layer commonly used in deep NNs to prevent overfitting, and is based on a regularization technique called stochastic regularization. The main idea is to look for certain parameters of the NN that try to learn about noisy patterns. This is performed by randomly dropping a part of the input neurons, around 20-50% of the neurons (varies depending on the number of neurons present in the layer). This attempts to detect which neurons are adding noise and which ones are actually adding value to the prediction.



## Exporting the Model

To export and reuse a model, we can make use a function called __SaveModel__, which is considered the standard for TensorFlow 2.x. It's officially recommended by the TensorFlow developer team as a "format" to share pre-trained models.

Fortunately, Keras hides a lot of the complexity from TensorFlow ([it's a bit more complicated to perform in TensorFlow than in Keras](https://www.tensorflow.org/guide/saved_model)) and allows us to do this very easily:

```python
my_first_model_path = './firstmodel/'
tf.saved_model.save(model, my_first_model_path)
# or: model.save(my_first_model_path)
```

Afterwards, we can reimport the model into another object after it's been saved:

```python
new_model = tf.saved_model.load(my_first_model_path)
```
```bash
>>> <tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject at 0x7f917c4ab7d0>
```


As we can see, working with TensorFlow isn't hard at all. In my personal opinion, TensorFlow seemed easier to use regarding loading/exporting data, and training models was also easier (thanks to using Keras and the TensorFlow ecosystem jointly). So, now is your time to begin. Head over to OCI and create a free account to get started! If you want to get started without spending a dollar, install a local notebook server in an OCI Compute machine and install TensorFlow! And if you run into trouble, you can always send me a message (I always appreciate messages and questions) and I'll help out as much as I can.

In the next articles we'll measure performance from both libraries, and focus on a performance + throughput analysis from both libraries when analyzing a dataset! It'll be insightful.
Stay tuned...

## How can I get started on OCI?

Remember that you can always sign up for free with OCI! Your Oracle Cloud account provides a number of Always Free services and a Free Trial with US$300 of free credit to use on all eligible OCI services for up to 30 days. These Always Free services are available for an **unlimited** period of time. The Free Trial services may be used until your US$300 of free credits are consumed or the 30 days has expired, whichever comes first. You can [sign up here for free](https://signup.cloud.oracle.com/?language=en&sourceType=:ow:de:te::::&intcmp=:ow:de:te::::).

## Join the conversation!

If you’re curious about the goings-on of Oracle Developers in their natural habitat, come [join us on our public Slack channel](https://join.slack.com/t/oracledevrel/shared_invite/zt-uffjmwh3-ksmv2ii9YxSkc6IpbokL1g?customTrackingParam=:ow:de:te::::RC_WWMK220210P00062:Medium_nachoLoL5)! We don’t mind being your fish bowl 🐠

## License

Written by [Ignacio Guillermo Martínez](https://www.linkedin.com/in/ignacio-g-martinez/) [@jasperan](https://github.com/jasperan), edited by [Erin Dawson](https://www.linkedin.com/in/dawsontech/)

Copyright (c) 2021 Oracle and/or its affiliates.

Licensed under the Universal Permissive License (UPL), Version 1.0.

See [LICENSE](https://github.com/oracle-devrel/leagueoflegends-optimizer/blob/main/LICENSE) for more details.
