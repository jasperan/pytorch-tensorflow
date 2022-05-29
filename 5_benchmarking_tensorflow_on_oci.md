# Benchmarking TensorFlow on OCI & EfficientNet's Models

## Introduction

Welcome to the fifth and last article in this series! We'll be discussing TensorFlow's efficiency with respect to PyTorch, something that we've already done [in the last article](https://github.com/jasperan/pytorch-tensorflow/blob/main/4_benchmarking_pytorch_on_oci.md).


## PerfZero

Since we already used the __pytorch-benchmark__ library last time, we're going to use another open-source library I mentioned called **PerfZero**. The main purpose is to execute TensorFlow tests to debug regression / classification performance. 

PerfZero makes it really easy to execute predefined tests.
Ideally, we want to use one of these three methods:
- Use PerfZero in a dedicated-infrastructure compute instance (with Docker)
- Use PerfZero locally on any computer (with Docker)
- Use PerfZero without Docker

The first option has the highest abstraction (as we're containerizing the program and all resources are virtualized) and the latest option has the lowest abstraction. We need to consider attempting to avoid interrupts, blocking calls and other types of exceptions / interrupts during the execution of our code in order to have an accurate benchmark with the lowest interruptions possible. It's up to you to decide which method to use. In this guide, I'm going to focus on reusing the resources we've already used in previous articles (our DS notebook sessions) to execute these tests. We will go through the steps to install PerfZero, Python, an initial virtual environment and the necessary packages to get started benchmarking TensorFlow on OCI.

## Creating an instance & setup on OCI

Here, we can choose to do two things: 
- Work with a Compute Instance from scratch, and install everything we need to get started.
- Work with a Data Science notebook session, which will help us with some things. Luckily for us, Data Science notebook sessions are deployed in a similar fashion as if it were a compute instance. This means that, through the use of CI/CD languages like Terraform, the deployment of a compute instance (together with several things inside of it) has been automated; so every time we create a DS notebook session we're basically creating a "superhero compute instance". 

This causes the underlying compute instance to have many things installed. This includes a default conda environment that we can access through the terminal:

![listing environments in conda](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/5-0.PNG?raw=true)

Conda, created by Anaconda, lets us use several Python environments interchangeably, so we can easily switch between our TensorFlow and PyTorch environments (as you can see in the picture above).

Since we're using TensorFlow in this article, let's switch to that environment:

```bash
conda activate /home/datascience/conda/tensorflow27_p37_cpu_v1
```

Now, let's check the Python version installed, as well as our TensorFlow version:

```bash
python --version
>>> Python 3.7.12

pip freeze | grep tensor
>>> tensorboard==2.7.0
>>> tensorboard-data-server==0.6.1
>>> tensorboard-plugin-wit==1.8.0
>>> tensorflow==2.7.0
>>> tensorflow-estimator==2.7.0
>>> tensorflow-io-gcs-filesystem==0.22.0
```

Looks like we're up to date with everything we need. Now, let's go ahead and install PerfZero. If you run into trouble running TensorFlow, please refer [to this article](https://github.com/jasperan/pytorch-tensorflow/blob/main/3_working_with_data_in_tensorflow.md) for an introduction to TensorFlow on OCI and how to get started.

We clone the repository into our machine:

```bash
git clone https://github.com/tensorflow/benchmarks.git
```

And we can build our Docker image to execute whenever we want. But before that, we need to install Docker into our machine. I used [this very handy](https://github.com/docker/docker-install) repository which has a root/non-root userscript to automatically install and setup the Docker daemon inside our machine:

```bash
curl -O https://github.com/docker/docker-install/blob/master/install.sh
# or
curl -O https://github.com/docker/docker-install/blob/master/rootless-install.sh
# execute whichever script you decided, depends on your machine permissions
# NOTE: if using DS notebook session from OCI, you'll need the rootless install
# then execute the Docker image build
python3 benchmarks/perfzero/lib/setup.py
```

After setting up our image, we can run the image in interactive mode (we attach to the shell process and can access the machine as if it were our own OS) by attaching a virtual volume into the data directory.

Note that [we can use any of the pre-trained models by TensorFlow](https://github.com/tensorflow/models/tree/master/official) (like we did in the last article with [EfficientNet](https://arxiv.org/abs/1905.11946)), as well as third-party community models. All of these models will yield a result when using PerfZero. 

Here's the command to follow:

```bash

nvidia-docker run -it --rm -v $(pwd):/workspace -v /data:/data perfzero/tensorflow \
python3 /workspace/benchmarks/perfzero/lib/benchmark.py --gcloud_key_file_url="" --git_repos="https://github.com/tensorflow/models.git;benchmark" --python_path=models --benchmark_methods=official.r1.resnet.estimator_benchmark.Resnet50EstimatorBenchmarkSynth.benchmark_graph_1_gpu --root_data_dir=/data
```

Note that you may run into some trouble with Google libraries. What I did to fix this is to modify the original code to ignore __gsutils__ and other libraries that were throwing errors during runtime. Note that the root of some of these issues is that the library was developed by the TensorFlow team, so they made integrations to automatically deploy in a Google Cloud instance instead of OCI. If you want to avoid these issues, you can [find the solution in the documentation](https://github.com/tensorflow/benchmarks/tree/master/perfzero#perfzero-on-local-workstation-or-any-server). The example that doesn't require accessing Google Cloud for any data will work in any instance.

And we can observe our model performance, in this case, a CIFAR-10 regressor, together with some model's metrics. Here's an example:

```json
 {
  "ml_framework_info": {                         # Summary of the machine learning framework
    "version": "1.13.0-dev20190206",             # Short version. It is tf.__version__ for TensorFlow
    "name": "tensorflow",                        # Machine learning framework name such as PyTorch
    "build_label": "ml_framework_build_label",   # Specified by the flag --ml_framework_build_label
    "build_version": "v1.12.0-7504-g9b32b5742b"  # Long version. It is tf.__git_version__ for TensorFlow
  },
  "execution_timestamp": 1550040322.8991697,     # Timestamp when the benchmark is executed
  "execution_id": "2022-05-25-02-41-42-133155",  # A string that uniquely identify this benchmark execution

  "benchmark_info": {                            # Summary of the benchmark framework setup
    "output_url": "gs://tf-performance/test-results/2022-05-25-02-41-42-133155/",     # Google storage url that contains the log file from this benchmark execution
    "has_exception": false,
    "site_package_info": {
      "models": {
        "branch": "benchmark",
        "url": "https://github.com/tensorflow/models.git",
        "hash": "f788046ca876a8820e05b0b48c1fc2e16b0955bc"
      },
      "benchmarks": {
        "branch": "master",
        "url": "https://github.com/tensorflow/benchmarks.git",
        "hash": "af9e0ef36fc6867d9b63ebccc11f229375cd6a31"
      }
    },
    "harness_name": "perfzero",
    "harness_info": {
      "url": "https://github.com/tensorflow/benchmarks.git",
      "branch": "master",
      "hash": "75d2991b88630dde10ef65aad8082a6d5cd8b5fc"
    },
    "execution_label": "execution_label"      # Specified by the flag --execution_label
  },

  "system_info": {                            # Summary of the resources in the system that is used to execute the benchmark
    "system_name": "system_name",             # Specified by the flag --system_name
    "accelerator_count": 2,                   # Number of GPUs in the system
    "physical_cpu_count": 8,                  # Number of physical cpu cores in the system. Hyper thread CPUs are excluded.
    "logical_cpu_count": 16,                  # Number of logical cpu cores in the system. Hyper thread CPUs are included.
    "cpu_socket_count": 1,                    # Number of cpu socket in the system.
    "platform_name": "platform_name",         # Specified by the flag --platform_name
    "accelerator_model": "Tesla V100-SXM2-16GB",
    "accelerator_driver_version": "410.48",
    "cpu_model": "Intel(R) Xeon(R) CPU @ 2.20GHz"
  },

  "process_info": {                           # Summary of the resources used by the process to execute the benchmark
    "max_rss": 4269047808,                    # maximum physical memory in bytes used by the process
    "max_vms": 39894450176,                   # maximum virtual memory in bytes used by the process
    "max_cpu_percent": 771.1                  # CPU utilization as a percentage. See psutil.Process.cpu_percent() for more information
  },

  "benchmark_result": {                       # Summary of the benchmark execution results. This is pretty much the same data structure defined in test_log.proto.
                                              # Most values are read from test_log.proto which is written by tf.test.Benchmark.report_benchmark() defined in TensorFlow library.

    "metrics": [                              # This is derived from `extras` [test_log.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/test_log.proto)
                                              # which is written by report_benchmark().
                                              # If the EntryValue is double, then name is the extra's key and value is extra's double value.
                                              # If the EntryValue is string, then name is the extra's key. The string value will be a json formated string whose keys
                                              # include `value`, `succeeded` and `description`. Benchmark method can provide arbitrary metric key/value pairs here.
      {
        "name": "accuracy_top_5",
        "value": 0.7558000087738037
      },
      {
        "name": "accuracy_top_1",
        "value": 0.2639999985694885
      }
    ],
    "name": "official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests.unit_test",    # Full path to the benchmark method, i.e. module_path.class_name.method_name
    "succeeded": true,                        # True iff benchmark method execution finishes without exception and no metric in metrics show succeeded = false
    "wall_time": 14.552583694458008           # The value is determined by tf.test.Benchmark.report_benchmark() called by the benchmark method. It is -1 if report_benchmark() is not called.
  }
}
```

## Measuring OCI Performance vs. GCE vs. AWS

Now that we've assessed how well TensorFlow and PyTorch can work on OCI, we should focus our attention on why OCI is the better solution for developing your own NN projects (or AI/ML for that matter).

### Virtual CPUs vs. OCPUs

Every cloud vendor often refers to their processing units as virtual CPUs, as the hardware is virtualized from the customer's perspective and we're given access to the machine and the corresponding resources "virtually". Also, several cloud vendors will talk about vCPUs as their measure of performance. One vCPU is equivalent to a [thread](https://www.geeksforgeeks.org/difference-between-process-and-thread/), which is what actually gets to run through our code (Python or not) and execute it sequentially.

![OCPUs with 2 threads](https://www.industry-era.com/images/article/OCPU.jpg?raw=true)

The problem is that many of these cloud vendors don't talk about threads, instead they talk about vCPUs. This is essential to understand OCI's advantage: an OCPU is equivalent to one physical core in the CPU unit, but it has **2 threads** instead of 1 (this is possible thanks to the hyper-threading technology developed by Intel, and it's quite hard to implement as you'd have to coordinate two threads' operations in the same processing unit without conflicts). So, by default, the number of threads contracted in our machines will be __double__ the amount of threads available in other vendors. This results in:
- Higher parallelization opportunities
- CPU load is decreased
- CPU scheduler works more, but not at all to the point of causing a bottleneck. (Should this ever happen, we can always increase resources dynamically in OCI!)


In case code couldn't be parallelized [(read what we've mentioned before about the GIL)](https://github.com/jasperan/pytorch-tensorflow/blob/main/1_getting_started_with_pytorch_on_oci.md#why-we-need-pytorch), having additional threads isn't necessarily beneficial; however TensorFlow and PyTorch are libraries that have been implemented with parallelization in mind and this is actually important for us, our benchmarks, and our models.

Of course, this applies to non-bare-metal instances in other cloud providers' websites, where the number of threads may be higher than 1; and it can be inferred that, as hyper-threading was created and designed by Intel, using a non-Intel processing unit in our compute instances will not yield the result explained in this section.

### Cost & Performance

If we compare OCI costs to, let's say, AWS, arguably the most popular cloud provider nowadays, we see huge differences in different sections:
- Networking: for every $1 charged at AWS, $0.26 are charged at Oracle.
- High performance computing (HPC): Oracle is about 44% cheaper
- Storage: Oracle is half as expensive for local SSDs
- RAM: Oracle is half as expensive
- Cold storage (Block storage): 2000% improvement in IOPS for 1/2 of the cost.

This means that the models we've previously created in this article series, and their metrics, can only be accounted for in OCI.

I've taken [this online tool](https://www.oracle.com/webfolder/workload-estimator/index.html) to generate a cost saver, if we were to deploy similar hardware infrastructure for our project, in AWS, and see how much we've saved (not to mention the rest of reasons mentioned above). If we recall, the last two articles were deployed with these characteristics:
- Intel(R) Xeon(R) Platinum 8167M @ 2.00GHz OCPUs (x16)

![OCI vs AWS](https://raw.githubusercontent.com/jasperan/pytorch-tensorflow/main/img/comp1.PNG?raw=true)


This'd cost us $588 in OCI, and a whooping **$1153** in AWS (49% saving percentage when coming into OCI) per month.

I hope this article series was as fun reading for you as it was for me to write it. Consider subscribing to our social media to stay tuned for future articles / cool projects, Hackathons and competitions with juicy benefits!

Stay tuned...


## How can I get started on OCI?

Remember that you can always sign up for free with OCI! Your Oracle Cloud account provides a number of Always Free services and a Free Trial with US$300 of free credit to use on all eligible OCI services for up to 30 days. These Always Free services are available for an **unlimited** period of time. The Free Trial services may be used until your US$300 of free credits are consumed or the 30 days has expired, whichever comes first. You can [sign up here for free](https://signup.cloud.oracle.com/?language=en&sourceType=:ow:de:te::::&intcmp=:ow:de:te::::).

## Join the conversation!

If you’re curious about the goings-on of Oracle Developers in their natural habitat, come [join us on our public Slack channel](https://bit.ly/devrel_slack)! We don’t mind being your fish bowl 🐠

## License

Written by [Ignacio Guillermo Martínez](https://www.linkedin.com/in/ignacio-g-martinez/) [@jasperan](https://github.com/jasperan), edited by [Erin Dawson](https://www.linkedin.com/in/dawsontech/)

Copyright (c) 2021 Oracle and/or its affiliates.

Licensed under the Universal Permissive License (UPL), Version 1.0.

See [LICENSE](https://github.com/oracle-devrel/leagueoflegends-optimizer/blob/main/LICENSE) for more details.
