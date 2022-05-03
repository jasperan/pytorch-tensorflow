# Working with Data in PyTorch

Welcome to the second article of this article series, where we explore the differences and similarities between PyTorch and TensorFlow, and how to work around data with both libraries.

In this article, we're going to do a deep-dive into PyTorch specifically, and how to work with data.

It's important to note that PyTorch is considered a Pythonic library, which means that it integrates very well with the Data Science stack already present in Python. It has some advantages over TensorFlow:
- It's newer than TensorFlow
- It's related to the Torch framework (a Lua ecosystem of community maintained packages for machine learning, parallel processing, data manipulation and computer vision) and the Python implementation renovated PyTorch's userbase
- It's used in Facebook by engineers
- It uses **tensors**, which can be thought of as a GPU-equivalent of NumPy arrays (or computer-optimized matrices).

![tensor](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/tensor.PNG?raw=true)

## PyTorch Fundamentals - Neural Networks

A neural network's (NN) implementation works just like a neuron in the human brain:
- We have artificial neurons called perceptrons
- A perceptron, just like a neuron would, connects with other neurons through axons (which in NNs are called **configurations**) to transmit data bilaterally

In NNs, perceptrons are composed of a series of inputs and produce an output. So, at least, we'll always have one input layer and one output layer; it's up to us programmers to decide how these layers communicate and in which order.

There are two types of neural networks:
- Feedforward NNs: data moves from the input layer to the output layer (through hidden layers or not, depends on the problem); and by the time data reaches the output layer, the NN has completed its job.
- Recurrent NNs: data doesn't stop at the output layer. Rather than doing so, it feeds again into previously-traversed layers recurrently, performing a specified number of cycles
    It's important to note that calculating gradients is based on [the chain rule](https://tutorial.math.lamar.edu/classes/calcI/ChainRule.aspx), which requires a bit of background in advanced mathematics. However, PyTorch has been kind enough to implement their own "automatic gradient calculator", called __autograd__, which does most of the mathematical work automatically. We'll talk more about this technique called [automatic differentiation](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/) later.

Here's an image of a feedforward NN, where we see only forward steps from the inputs (below) towards the outputs (above):

![feedforward](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/feedforward.PNG?raw=true)

And here's an image of a recurrent NN. Note that if we have more than one hidden layer, we can call the NN a **deep NN**.

![recurrent](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/recurrent.PNG?raw=true)

## Tensors

In PyTorch, we have **tensors**. As mentioned in the previous article, a PyTorch tensor is exactly equivalent as a NumPy array. Just like with the numpy library, operating with a tensor will allow us to perform optimized operations to the data we have. It's fairly common to encounter a PyTorch tensor being executed by the GPU (as GPUs have many more processing units than the CPU itself), although it's a common misconception to believe that it can't be executed in a CPU.

## Loading Data

To load data, we can follow several steps:
- Loading a dataset from a Python compatible standard library, like reading from a CSV file using *pandas*, and then converting the dataset into a tensor.
- Using the __torchvision__  package, which allows us to load and prepare a dataset using a multiprocessing implementation.

In the spirit of showing how PyTorch works, we're going to load a [famous dataset](https://image-net.org/) from a list of available pre-loadable, built-in datasets:

### Built-in Datasets
```python
import argparse
# we get the number of threads we want
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threads', help='Number of threads to use', required=True)
args = parser.parse_args() 

# we load the image net dataset
imagenet_data = torchvision.datasets.ImageNet('./imagenet/')

# By using the DataLoader function, we can load the dataset in n batch sizes (by default 1) 
# shuffle is used to randomize the dataset's row order
data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True, num_workers=args.threads)
```

You can find the list of built-in datasets [here](https://pytorch.org/vision/stable/datasets.html).

## Automatic Differentiation in PyTorch

The complex thing about Neural Networks is how they are constructed. If we're getting started, it'd be pretty complex taking an intensive course in differential equations and integrals in order to understand Neural Networks. 

Automatic differentiation is a technique implemented by PyTorch (which didn't exist before, which made Neural Networks a very mathematically-complex topic that forced people to calculate gradients on Recurrent NNs)
## Performing Linear Regression

We're going to perform one of the simplest regression tasks in ML with PyTorch: linear regression.

First, we import all the packages we need to generate artificial data. For that, we're going to use numpy's random function:

```python
import numpy as np
import pandas as pd

# we define the 3 components of linear regression
# y = mx + b (m = slope, b = intercept)
# y is what we need to predict (dependent variable)
m = 1.5
b = 3

x = np.random.rand(1024)
randomness = np.random.rand(1024) / 2 # adding some randomness.
y = m*x + b + randomness # we have our linear regression formula with the added randomness

# create an empty dataframe and populate it.
df = pd.DataFrame()
df['x'] = x
df['y'] = y
```

We plot the current random data:

```python
import seaborn as sns

sns.lmplot(x = 'x', y = 'y',
    data = df)
```

![8](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/8.PNG?raw=true)

And we proceed to create a Neural Network able to predict the target variable 'y'. We have this data already in the __df__ dataframe. Note that, as mentioned before, we're able to make a conversion between a __pandas__ object and a PyTorch object.

```python
import torch
import torch.nn as nn
from torch.autograd import Variable # we import autograd for automatic differentiation
# as mentioned in the last article, we create our object with an init constructor,
# and the forward function to define the NN's configuration. In this case, we'll just have one step from the 
# input layer to the output layer (linear).
class LinearRegressorNN(nn.Module):

    def __init__(self, input_dim, output_dim):
      super(LinearRegressionModel, self).__init__()
      # linear equation is of the form Wx = B where W is a weight, x is the input and B is the output.
      # it's the simplest form of PyTorch
      self.linear = nn.Linear(input_dim, output_dim) # nn.Linear is required for linear regression

   def forward(self, x):
      out = self.linear(x)
      return out
```

As a reminder, here's a depiction of the data format we've stored inside the dataframe __df__:

![9](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/9.PNG?raw=true)

As we just have one feature and one target variable (feature being x, the independent variable, and target being y, the dependent variable), we define our dimensions as:

```python
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
# we create an object of our above class
model = LinearRegressorNN(input_dim, output_dim)

# the loss function will ultimately determine how gradients are calculated.
# in recurrent NNs, gradients are computed by applying the chain rule from the loss function backwards.
criterion = nn.MSELoss() # we define our loss function as the mean squared error
[w, b] = model.parameters()
print(w, b)
```

This would be categorized as a **feedforward linear regression NN**, as we're not calculating any gradients or using autograd.

