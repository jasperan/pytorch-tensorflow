# Benchmarking PyTorch on OCI

Welcome to the fourth article on this series where we do a deep dive into Neural Networks as a whole. In this specific article, we're going to talk about PyTorch's efficiency and performance when being challenged with different parameters, and how these parameters can affect the training time of a model.

From a popularity perspective, I extracted this information from Google Trends to analyze both popularities:

![1](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/1.PNG?raw=true)

As we can observe, TensorFlow is reigning right now over the world. Let's see if performance matches expectations.

## Machine Specifications

For conducting these tests, we need to make sure that the hardware in which we test has the same specifications as in the next article, where we'll analyze TensorFlow's performance.

Additionally, it's recommended to perform tests in private/dedicated infrastructure, meaning that we'd ideally need:
- Dedicated infrastructure instead of using shared infrastructure. This means that it'd be ideal to have an OCI Compute Instance being given to you and noone else. This prevents the CPU, for example, to virtually share resources with other customers signed up in OCI. This will in turn reduce I/O interruptions and such, which will make our benchmarking tests much more accurate.
- The same notebook session, so that the Operating System doesn't accidentally give more priority to one Jupyter / Zeppelin project than the other through the CPU scheduler.

The machine specifications for this test are:

----------------------- TODO -----------------------------------

## Using EfficientNet models

For measuring PyTorch's performance, we'll use [a NN called EfficientNet](https://arxiv.org/abs/1905.11946). It's a pre-trained convolutional neural network that attempts to systematically change how people approach design & architecture of their own models. In this case, EfficientNet focuses on applying its own specific scaling throughout all dimensions of an image (depth, width, resolution) using a coefficient called C. Using this coefficient has been proven to work pretty well (one of its variants, called EfficientNet 7 has the highest accuracy ever on the ImageNet dataset), being about 8 times smaller than other models in the top list; and being about 6 times quicker. Aside from being able to predict results accurately against the ImageNet dataset (which could be considered an extremely difficult task by itself), it performs very well on other commonly known datasets such as [CIFAR100, a dataset that contains images of animals](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/g3doc/flops.png), the Flowers dataset and three other datasets; with many less parameters than the model's competitors.

Think of EfficientNet models as the automation of the design & architecture part of NNs, which is very helpful, not only for our specific use case, but in several image processing problems that you'll find in your career as a Data Scientist / Data Analyst and similar.




https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/g3doc/params.png
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/g3doc/flops.png


EfficientNet uses millions of parameters, and can therefore be considered a deep Neural Network. I wanted to mention the effect that the Dropout layer (which we applied in last article) can have on significantly improving the accuracy of an NN. Here, we can graphically observe the effect of a Dropout layer when implemented, by looking at which neurons are activated and which ones are deactivated, and how this hidden layer implementation attempts to reduce the noise created by inaccurate predictions, subsequently increasing the train/test accuracy of the model:

![mnist1](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/mnist1.PNG?raw=true)

![mnist2](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/mnist2.PNG?raw=true)

One can argue that EfficientNet is a "simple" model, but in reality we need to consider that, since training is done automatically by computers, millions of operations are performed every second, which causes models like this to have millions of parameters. I took some visualizations from the original TensorFlow repository (which has some documentation about EfficientNet) and displayed it here for you, so you can have a sense of "size" of models such as this one; and to give you a perspective of how well EfficientNet's new coefficient C is compared to other models.

The first image considers the number of parameters in millions, compared to other models; and the second one compares the number of floating point operations performed by the models to make predictions.

![vis1](https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png?raw=true)

![vis2](https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png?raw=true)



## Benchmarking tools available

There are several options to perform a benchmark. Of course, we can always use the standard libraries offered by Python to help us with this, or choose a more advanced approach like [PerfZero](https://github.com/tensorflow/benchmarks/tree/master/perfzero) or similar tools. In this case, we're going to avoid complicated libraries, as learning how to perform a benchmark in a correct way is more important than learning how to use a specific library / tool. As technology changes, I'll always say the most important thing is to have the concepts and basic ideas in our minds, and then dwelve into exploring and trying specifics if we need them in our use cases.

So, to create this article I've tested a very easy to use package that I found from PyPi called [pytorch-benchmark](https://pypi.org/project/pytorch-benchmark/), and a standard time measurement. Props to [Lukas Hedegaard](https://github.com/LukasHedegaard/pytorch-benchmark) for the great package.

First of all, we install the library:

```bash
pip install pytorch-benchmark
```

We load a MNIST model into our code:

```python

```

And we use the library to help us measure performance.



## Measuring Performances

There are several metrics being considered behind the scenes. A model's performance can be measured with a combination of these factors:
-
-
-
-
-
-


## Conclusions

As we can see, it wasn't so easy to benchmark the performance


## How can I get started on OCI?

Remember that you can always sign up for free with OCI! Your Oracle Cloud account provides a number of Always Free services and a Free Trial with US$300 of free credit to use on all eligible OCI services for up to 30 days. These Always Free services are available for an **unlimited** period of time. The Free Trial services may be used until your US$300 of free credits are consumed or the 30 days has expired, whichever comes first. You can [sign up here for free](https://signup.cloud.oracle.com/?language=en&sourceType=:ow:de:te::::&intcmp=:ow:de:te::::).

## Join the conversation!

If you’re curious about the goings-on of Oracle Developers in their natural habitat, come [join us on our public Slack channel](https://join.slack.com/t/oracledevrel/shared_invite/zt-uffjmwh3-ksmv2ii9YxSkc6IpbokL1g?customTrackingParam=:ow:de:te::::RC_WWMK220210P00062:Medium_nachoLoL5)! We don’t mind being your fish bowl 🐠

## License

Written by [Ignacio Guillermo Martínez](https://www.linkedin.com/in/ignacio-g-martinez/) [@jasperan](https://github.com/jasperan), edited by [Erin Dawson](https://www.linkedin.com/in/dawsontech/)

Copyright (c) 2021 Oracle and/or its affiliates.

Licensed under the Universal Permissive License (UPL), Version 1.0.

See [LICENSE](https://github.com/oracle-devrel/leagueoflegends-optimizer/blob/main/LICENSE) for more details.
