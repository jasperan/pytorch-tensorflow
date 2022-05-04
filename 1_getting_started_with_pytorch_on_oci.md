# Getting Started with PyTorch on OCI

Welcome to the first article of this series, where we’ll explore AI/ML libraries such as PyTorch and TensorFlow. Typically, there’s a positive sentiment towards PyTorch and how great it is, and a more negative social sentiment towards TensorFlow. However, both libraries have unique capabilities and I hope that, through this series of articles, I’ll be able to break the stigma and show that both libraries are great while showing some of the capabilities along the way.

We’ll learn how to set up OCI to work on an issue with PyTorch, how to develop this issue into a solution with a robust architecture, and finally benchmark the performance of both libraries to see which library works best for that specific issue.

## A Brief History of ML

Before getting started on setting up environments and all the "technical" stuff, there are couple of things to take note on the history of these libraries and Python itself:
- Python was created 30 years ago, something that most people wouldn't believe as this programming language "exploded" in popularity not so long ago. 
- TensorFlow was created in 2015 by Google
- PyTorch was created a year after by FAIR (Facebook's AI Research laboratory)

As Tensorflow was created a year before, it gained popularity quickly in the Data Science world, and this increase can be observed in the general sentiment of these libraries, as well as by comparing the number of commits in both open-source repositories, where [TensorFlow](https://github.com/tensorflow/tensorflow) has about 3 times the number of stars in GitHub than [PyTorch's](https://github.com/pytorch/pytorch).

It’s clear that nowadays, Data Science is growing very rapidly since we have an ever-increasing amount of unstructured (and structured!) data available to us. It’s our job to understand this data and make sense of it. In the second half of the 20th century, Artificial Intelligence (AI) as we know it today was created. This allowed humans to “relax” their complex calculations, while delegating this job to a machine. Machine Learning has since become an important part of everyday life, even if it’s not apparent:
- Email spam filters are based on ML models
- Advanced video-game anti cheating systems (to prevent hackers) are based on ML models, that compare data from legitimate players to data from cheaters and analyzes divergence to determine unfair play
- Netflix and Amazon Prime’s suggestions for what to watch are based on a ML model that analyzes your taste and makes similar recommendations
- Tesla’s autopilot driving software is based on computer vision and ML models that make real-time decisions on driving in society

## Why we need PyTorch

PyTorch is a great library that synergizes very well with Python. When performing data analysis with Python, we need to understand that Python’s interpreter is limited to execute on only one processor. This is called the GIL or Global Interpreter Lock, a mutex that allows only one thread to execute the interpreter (this can be avoided by implementing code with the **multiprocessing module**). However, most household computers and non-professional equipment rarely come with more than 16/32 cores, which means that we can theoretically improve the code optimization by 16/32-fold at maximum unless we use the GPU to help us.

Luckily for us, PyTorch is GPU friendly: we can execute our code in CPUs, GPUs and even TPUs, Tensor Processing Units, which is a specific unit of hardware developed by Google designed to be used for AI/ML purposes mostly.

From a mathematical perspective, a tensor is a group of data. A number is equivalent to a rank-0 tensor; a 1-dimensional array (vector) is a rank-1 tensor and a matrix is a rank-2 tensor. This goes on and on for more dimensions until we get a rank-n tensor.

Tensors allow us to group our data into optimized subsets that will run efficiently in our hardware.

## Getting Started

First, we need an environment where we’ll run PyTorch code. For that, we head to the Oracle Cloud Infrastructure console.

We have two options:

- Create a compute instance, spin it up, install Jupyter Notebook or other notebook software where we’ll run our Python code, and install PyTorch. This takes a bit longer than the second option since we have to do all the configuration ourselves, however we’ll save some money in the long run as creating a compute instance is a bit cheaper than the second option.
- Create a OCI Data Science notebook and install PyTorch. This is very straightforward and doesn’t require that much IT knowledge; we’ll have a live visualization of our notebooks and an interface to modify them in our browser.

In this subsection, I'll showcase how to follow the second option (OCI Data Science).

### Create an OCI Data Science Notebook and Install PyTorch

First, I head over to the OCI console and navigate to OCI Data Science:

![1](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/1.PNG?raw=true)

Secondly, I create a new project:

![2](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/2.PNG?raw=true)

Inside this project, we can have several notebook sessions; and these sessions will have their own respective storage each. Also, notebook sessions can be collaborative and edited by multiple OCI users concurrently.

![3](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/3.PNG?raw=true)

Now that we’re inside the notebook, we have control over the machine (we can access it through a Terminal just like we would if we ssh’d into the machine) or we can control Python environments through the Environment Explorer. For new users, I highly recommend the environment explorer, as it has several pre-built environments ready to go. We can find PyTorch in the environment explorer very easily, and install it.

![4](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/4.PNG?raw=true)

To install it, we run the command with the pre-built identifier in a terminal:

![5](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/5.PNG?raw=true)

![6](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/6.PNG?raw=true)


Whichever section you decide to go for, we'll have to install PyTorch. For that, let's follow [these steps](https://pytorch.org/get-started/locally/). I personally recommend **conda** as the package manager as it eases the manipulation of virtual environments:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Or the equivalent with pip:

```bash
pip3 install torch torchvision torchaudio
```

Once we have PyTorch installed, we can check that a test notebook runs smoothly in our notebook. For that, we can run the pre-defined list of notebook examples (that were automatically installed together with the PyTorch environment from the Environment Explorer) or run an example ourselves. We will do a mix of both to test the functionality of PyTorch using the [very well-known iris dataset](https://gist.github.com/curran/a08a1080b88344b0c8a7).

We load the iris dataset:
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F

iris = datasets.load_iris()
X = iris['data'] # dependent variables or features
y = iris['target'] # independent variable or target

scaler = StandardScaler() # we scale our data for normalization purposes as features don't follow a normal distribution (e.g. the sepal length is about 10-20 times bigger than the petal width, both of them being features of the model)
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2) # we split our data 80/20%
```

We configure a very simple Neural Network model with 3 linear layers and [Adam optimization](https://towardsdatascience.com/complete-guide-to-adam-optimization-1e5f29532c3d). If you’re unfamiliar with these concepts, don’t worry, we’ll have an in-depth look into what these things mean in the following articles from this series. For now, just know that Adam is an optimization algorithm with a complexmathematical formula (see the image right below), but to get started, we don’t need to fixate on this. You just need to know the following about Adam optimization (in general):
- It’s easy to implement
- It’s memory efficient
- It’s good for data intensive problems, which is why Adam optimization is well-known in Big Data
- Hyper-parametrization (model tuning) is overshadowed by accurate results yielded by the model, which means, we’ll generaly save some time.

![adam](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/adam.PNG?raw=true)

```python
# we create a neural network 
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        # with 3 linear layers
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    # it is compulsory to define the forward function.
    # this function will pass data into the computation graph of the NN
    # and will represent the algorithm.
    # we can use any of the tensor operations inside the forward function, like relu and softmax.
    def forward(self, x):
        # ReLU is the activation function that makes the Neural Network non-linear.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1) # our output layer will be a softmax layer
        # otherwise we wouldn't be able to interpret the result as easily
        return x


model = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
```

Basically, we will have 3 inputs and the result of the Neural Network will be something like this:

![7](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/7.PNG?raw=true)

And we run the code for 100 epochs. An epoch means that the "training" process we've been doing in previous articles, is repeated n times, where n is an integer bigger than 0.

```python
EPOCHS = 100
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in tqdm.trange(EPOCHS):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()
    
    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()
```

Now, we can create a [model artifact](https://docs.oracle.com/en-us/iaas/data-science/using/manage-models.htm). This is especially useful when developing code using Oracle Data Science notebooks as it integrates with Oracle ADS (Accelerated Data Science) which simplifies saving and reusing the model in the future with simple commands.

```python
# we create the artifact in a temporary directory and store it in a pickle file, like in previous articles
# Local path where the artifact will be stored.
model_artifact_path = mkdtemp()

# preparing the model artifact in a local directory: 
model_artifact = prepare_generic_model(model_artifact_path,
    data_science_env=True,
    force_overwrite=True)

# saving the PyTorch model in the same model artifact directory: 
torch.save(model,
    os.path.join(model_artifact_path,
        'torch_lr.pkl'))

print(f"The model artifact is stored in: {model_artifact_path}")
```
This returns:
```bash
>>> The model artifact is stored in: /tmp/tmp3sdx9i2r
```

And now we can access the model artifact saved in the temporary directory and make a test prediction:

```python
test_data = torch.tensor(X_test[:10].tolist())
model_artifact.predict(test_data)
```

```bash
>>> {'prediction': [[0.9936151504516602,
   0.0057501643896102905,
   0.0006346192094497383]]}
```

As we previously defined a rank-1 tensor (vector) beforehand, we are returned with three different numbers. If we sum them, it yields 1; and each one represents the probability of each sample to be a given species.

A note on that probability: it’s actually the weight of the categorization performed by the Neural Network by the activation function, which isn’t technically a probability being returned; but it basically means that the bigger the number in the vector, the better chances the NN will decide that its category is the one in that position of the array.

I really hope that you enjoyed reading and learning about how to get started with PyTorch on OCI.

## How can I get started on OCI?

Remember that you can always sign up for free with OCI! Your Oracle Cloud account provides a number of Always Free services and a Free Trial with US$300 of free credit to use on all eligible OCI services for up to 30 days. These Always Free services are available for an **unlimited** period of time. The Free Trial services may be used until your US$300 of free credits are consumed or the 30 days has expired, whichever comes first. You can [sign up here for free](https://signup.cloud.oracle.com/?language=en&sourceType=:ow:de:te::::&intcmp=:ow:de:te::::).

## Join the conversation!

If you’re curious about the goings-on of Oracle Developers in their natural habitat, come [join us on our public Slack channel](https://join.slack.com/t/oracledevrel/shared_invite/zt-uffjmwh3-ksmv2ii9YxSkc6IpbokL1g?customTrackingParam=:ow:de:te::::RC_WWMK220210P00062:Medium_nachoLoL5)! We don’t mind being your fish bowl 🐠

## License

Written by [Ignacio Guillermo Martínez](https://www.linkedin.com/in/ignacio-g-martinez/) [@jasperan](https://github.com/jasperan), edited by [Erin Dawson](https://www.linkedin.com/in/dawsontech/)

Copyright (c) 2021 Oracle and/or its affiliates.

Licensed under the Universal Permissive License (UPL), Version 1.0.

See [LICENSE](https://github.com/oracle-devrel/leagueoflegends-optimizer/blob/main/LICENSE) for more details.