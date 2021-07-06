import importlib
#from natsort import natsorted

import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
from Dataset import Dataset


def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5) # <--

    outputs = []

    transform = transforms.Compose([#transforms.Resize((256, 256)),
                                    transforms.ToTensor()
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                   ])

    train_dataset=Dataset("dataset1", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)


    # Loop over epochs
    for epoch in range(num_epochs):

        # Training
        for idx, img in enumerate(train_loader):
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("here again")

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs


def main():
    module = importlib.import_module('Autoencoder')
    model = module.Autoencoder()
    max_epochs = 20
    outputs = train(model, num_epochs=max_epochs)


if __name__ == '__main__':
    main()


