import argparse
from scipy.stats import norm, bernoulli
import numpy as np 
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.hidden_dimension = hidden_dim
        self.z_dimension = z_dim
        self.input_dim_hidden = nn.Linear(784, hidden_dim) # 784 input dimension of MNIST data image
        self.hidden_mean = nn.Linear(hidden_dim, z_dim)
        self.hidden_std = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        hidden_input = nn.functional.relu(self.input_dim_hidden(input)) #original paper uses tanh activation functions
        mean, std = self.hidden_mean(hidden_input), self.hidden_std(hidden_input)
        assert mean.shape == (input.shape[0], self.z_dimension)
        assert std.shape == (input.shape[0], self.z_dimension)
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.z_hidden_vector = nn.Linear(z_dim, hidden_dim)
        self.hidden_linear = nn.Linear(hidden_dim, 784) # 784 input dimension of MNIST data image

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        self.transformed_input = nn.functional.relu(self.z_hidden_vector(input)) 
        self.transformed_input_ = self.hidden_linear(self.transformed_input)

        mean = torch.sigmoid(self.transformed_input_)
        assert mean.shape == (input.shape[0], 784)
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)


    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        #Encoding step
        mean, std = self.encoder(input)
        epsilon = torch.randn_like(mean)
        z = mean + torch.exp(0.5 * std) * epsilon
        
        #Decoding step
        encoded_sample = self.decoder(z)

        #Reconstruction term
        l_recon = nn.functional.binary_cross_entropy(encoded_sample, input)
        
        #Regularised term 
        #This is different than my own derivation but is similar and its the same as the original paper, where there the use the '-' 
        l_reg_KL = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp()) /input.shape[0]

        average_negative_elbo = (l_recon + l_reg_KL) 

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        #  Create meshgrid of n_samples by z_dim
        # TODO make accept more than two dimensions
        sample_z = torch.from_numpy(norm.ppf(np.meshgrid(np.linspace(1e-9, 0.99, n_samples)))).float()
        
        im_means = self.decoder(sample_z)
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0.0
    for step, x_batch in enumerate(data):
        x_batch = x_batch.view(-1, 784)

        if model.training:
            optimizer.zero_grad()
            elbo = model(x_batch)
            elbo.backward()
            optimizer.step()
        else:
            elbo = model(x_batch)
        average_epoch_elbo += elbo.item()

    return average_epoch_elbo/len(data)


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        sampled_img = model.sample(10)[0]
        sampled_img = sampled_img.view(-1, 28, 28)
        grid = make_grid(sampled_img, nrow=2)
        plt.imsave('VAE_EPOCH' + str(epoch) +'.png', grid.permute(1, 2, 0).detach().numpy())
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
