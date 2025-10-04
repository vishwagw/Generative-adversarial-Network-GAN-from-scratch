# Handwritten digits generator GAN from scratch

GANs are being used for high-dimensional data such as images and this project is to demonstrate the capabiities of GAN.

We are using torchvision package and its MNIST dataset. 

Again, you’re using a specific version of torchvision to assure the example code will run, just like you did with pytorch. With the environment set up, you can start implementing the models in Jupyter Notebook. Open it and create a new Notebook by clicking on New and then selecting gan.

## CPU and GPU computational power: 

Since this example uses images in the training set, the models need to be more complex, with a larger number of parameters. This makes the training process slower, taking about two minutes per epoch when running on CPU. You’ll need about fifty epochs to obtain a relevant result, so the total training time when using a CPU is around one hundred minutes.

To reduce the training time, you can use a GPU to train the model if you have one available. However, you’ll need to manually move tensors and models to the GPU in order to use them in the training process.

## preparing dataset: 

The MNIST dataset consists of 28 × 28 pixel grayscale images of handwritten digits from 0 to 9. To use them with PyTorch, you’ll need to perform some conversions.

The original coefficients given by transforms.ToTensor() range from 0 to 1, and since the image backgrounds are black, most of the coefficients are equal to 0 when they’re represented using this range.

transforms.Normalize() changes the range of the coefficients to -1 to 1 by subtracting 0.5 from the original coefficients and dividing the result by 0.5. With this transformation, the number of elements equal to 0 in the input samples is dramatically reduced, which helps in training the models.

The arguments of transforms.Normalize() are two tuples, (M₁, ..., Mₙ) and (S₁, ..., Sₙ), with n representing the number of channels of the images. Grayscale images such as those in MNIST dataset have only one channel, so the tuples have only one value. Then, for each channel i of the image, transforms.Normalize() subtracts Mᵢ from the coefficients and divides the result by Sᵢ.

Aftet downloading, we can create the train dataset data loader.

We can use Matplotlib to plot some samples of the training data.

## Discriminator and Generator implementaion:

In this case, the discriminator is an MLP neural network that receives a 28 × 28 pixel image and provides the probability of the image belonging to the real training data.

Since the generator is going to generate more complex data, it’s necessary to increase the dimensions of the input from the latent space. In this case, the generator is going to be fed a 100-dimensional input and will provide an output with 784 coefficients, which will be organized in a 28 × 28 tensor representing an image.

## Training models: 

To train the models, you need to define the training parameters and optimizers.

To obtain a better result, you decrease the learning rate from the previous example. You also set the number of epochs to 50 to reduce the training time.

then we must create a training loop. 

Some of the tensors don’t need to be sent to the GPU explicitly with device. This is the case with generated_samples in line 11, which will already be sent to an available GPU since latent_space_samples and generator were sent to the GPU previously.

Since this example features more complex models, the training may take a bit more time. After it finishes, you can check the results by generating some samples of handwritten digits.

## Checking samples:

To generate handwritten digits, you have to take some random samples from the latent space and feed them to the generator.

Then we can plot the generated results:
 you need to move the data back to the CPU in case it’s running on the GPU. For that, you can simply call .cpu(). As you did previously, you also need to call .detach() before using Matplotlib to plot the data.


## endnotes:

despite the complexity of GANs, machine learning frameworks like PyTorch make the implementation more straightforward by offering automatic differentiation and easy GPU setup.

