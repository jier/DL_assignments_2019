import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.BatchNorm1d(784),
            nn.Tanh()
        )
    def forward(self, z):
        # Generate images from z
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(784, 512), 
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.layers(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):
    # TODO Make this also run on LISA
    generator_loss = []
    discriminator_loss = []
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            if torch.cuda.is_available():
                imgs.cuda()
            data_img  = imgs.view(-1, 784).to(device)
            batch_size = data_img.shape[0]

            # Train Generator
            # ---------------
            latent_z = torch.randn(batch_size, args.latent_dim).to(device)
            fake_img = generator(latent_z).to(device)

            decision_discrmntor = discriminator(fake_img).to(device)
            loss_gen = - torch.log(decision_discrmntor).sum() # could also use binary cross entropy as data is binary

            optimizer_G.zero_grad()
            loss_gen.backward()
            torch.nn.utils.clip_grad_norm(generator.parameters(), max_norm=10)
            optimizer_G.step()
            # Train Discriminator
            # -------------------
            latent_z = torch.randn(batch_size, args.latent_dim).to(device)
            fake_img = generator(latent_z).to(device)
            fake = discriminator(fake_img).to(device)
            real = discriminator(data_img).to(device)

            loss_dscr = -(torch.log(real) + torch.log(1 - fake)).sum()
            optimizer_D.zero_grad()
            loss_dscr.backward()
            torch.nn.utils.clip_grad_norm(discriminator.parameters(), max_norm=10)
            optimizer_D.step()

            generator_loss.append(loss_gen.item())
            discriminator_loss.append(loss_dscr.item())
            # Does not change??
            positives = (real > 0.5).sum().item()
            negatives = (fake <= 0.5).sum().item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}|{args.n_epochs} Enum dataloader{i} Loss G {loss_gen} Loss D {loss_dscr} Positive {positives} and Negatives {negatives}")

            #TODO what to do if discriminator is too good

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0 or ((epoch == args.n_epochs-1) and (i == len(dataloader)-1)):
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(fake_img[:25].view(-1, 1, 28, 28),
                           'images/{}_{}.png'.format(epoch, batches_done),
                           nrow=5, normalize=True)

def interpolate(device):

    checkpoint = torch.load('mnist_generator.pt', map_location=torch.device(device))
    generator = Generator(args.latent_dim)
    generator.load_state_dict(checkpoint)
    generator.eval()

    latent_z = torch.randn(2, args.latent_dim).to(device)
    # Add dimension to work as input to interpolate function
    temp_z = torch.T(latent_z).unsqueeze(0)
    # linear interpolate vectors
    im_interp = nn.functional.interpolate(temp_z, mode='linear', size=9, align_corners=True)
    interp_img = generator(im_interp).to(device)
    save_image(interp_img.view(-1, 1, 28, 28), 'images/interpolated.png', nrow=9, normalize=True)



def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device=device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")
    interpolate(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
