# GAN from scratch:

For creating the sample GAN, we are using pytorch suite with libraries including matplotlib and numpy.

## image transformation:
For this GAN, we are using CIDAR-10 dataset.
Download and load the CIFAR-10 dataset with defined transformations. Use a DataLoader to process the dataset in mini-batches of size 32 and shuffle the data.

## Defining GAN's parameters:
Set important training parameters:

 * latent_dim: Dimensionality of the noise vector.
 * lr: Learning rate of the optimizer.
 * beta1, beta2: Beta parameters for Adam optimizer (e.g 0.5, 0.999)
 * num_epochs: Number of times the entire dataset will be
 processed (e.g 10)

## designing the generator model:
Create a neural network that converts random noise into images. Use transpose convolutional layers, batch normalization and ReLU activations. The final layer uses Tanh activation to scale outputs to the range [-1, 1].

## designing the discriminator model:

Create a binary classifier network that distinguishes real from fake images. Use convolutional layers, batch normalization, dropout, LeakyReLU activation and a Sigmoid output layer to give a probability between 0 and 1.

## initializing the GAN models:
Generator and Discriminator are initialized on the available device (GPU or CPU).
Binary Cross-Entropy (BCE) Loss is chosen as the loss function.
Adam optimizers are defined separately for the generator and discriminator with specified learning rates and betas.

## Training the GAN:

Train the discriminator on real and fake images, then update the generator to improve its fake image quality. Track losses and visualize generated images after each epoch.

valid = torch.ones(real_images.size(0), 1, device=device): Create a tensor of ones representing real labels for the discriminator.
fake = torch.zeros(real_images.size(0), 1, device=device): Create a tensor of zeros representing fake labels for the discriminator.
z = torch.randn(real_images.size(0), latent_dim, device=device): Generate random noise vectors as input for the generator.
g_loss = adversarial_loss(discriminator(gen_images), valid): Calculate generator loss based on the discriminator classifying fake images as real.
grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True): Arrange generated images into a grid for display, normalizing pixel values.



