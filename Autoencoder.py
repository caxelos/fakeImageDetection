import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self,args):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(in_channels=3, out_channels=args.channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.channels[0], out_channels=args.channels[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.channels[1], out_channels=args.channels[2], kernel_size=7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=args.channels[2], out_channels=args.channels[1], kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=args.channels[1], out_channels=args.channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=args.channels[0], out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
