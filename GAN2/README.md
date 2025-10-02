# GAN - 2 from scratch:

Source: https://realpython.com/generative-adversarial-networks/ 

As a first experiment with generative adversarial networks, you’ll implement the example described in the previous section.

## Data preprocessing and creating two NN models: 

To run the example, you’re going to use the PyTorch library, which you can install using the Anaconda Python distribution and the conda package and environment management system.
To begin, create a conda environment and activate it:

$ conda create --name gan
$ conda activate gan

After you activate the conda environment, your prompt will show its name, gan. Then you can install the necessary packages inside the environment:

$ conda install -c pytorch pytorch=1.4.0
$ conda install matplotlib jupyter

Since PyTorch is a very actively developed framework, the API may change on new releases. To ensure the example code will run, you install the specific version 1.4.0.

Besides PyTorch, you’re going to use Matplotlib to work with plots and a Jupyter Notebook to run the code in an interactive environment. Doing so isn’t mandatory, but it facilitates working on machine learning projects.

For a refresher on working with Matplotlib and Jupyter Notebooks, take a look at Python Plotting With Matplotlib (Guide) and Jupyter Notebook: An Introduction.

Before opening Jupyter Notebook, you need to register the conda gan environment so that you can create Notebooks using it as the kernel. To do that, with the gan environment activated, run the following command:

$ python -m ipykernel install --user --name gan

Now you can open Jupyter Notebook by running jupyter notebook. Create a new Notebook by clicking New and then selecting gan.

Here, you import the PyTorch library with torch. You also import nn just to be able to set up the neural networks in a less verbose way. Then you import math to obtain the value of the pi constant, and you import the Matplotlib plotting tools as plt as usual.

It’s a good practice to set up a random generator seed so that the experiment can be replicated identically on any machine.

## Preparing data: 
Here, you compose a training set with 1024 pairs (x₁, x₂). In line 2, you initialize train_data, a tensor with dimensions of 1024 rows and 2 columns, all containing zeros. A tensor is a multidimensional array similar to a NumPy array.

In line 3, you use the first column of train_data to store random values in the interval from 0 to 2π. Then, in line 4, you calculate the second column of the tensor as the sine of the first column.

Next, you’ll need a tensor of labels, which are required by PyTorch’s data loader. Since GANs make use of unsupervised learning techniques, the labels can be anything. They won’t be used, after all.

In line 5, you create train_labels, a tensor filled with zeros. Finally, in lines 6 to 8, you create train_set as a list of tuples, with each row of train_data and train_labels represented in each tuple as expected by PyTorch’s data loader.


## data training:

For training both models, we  are using an adams algorithm.
