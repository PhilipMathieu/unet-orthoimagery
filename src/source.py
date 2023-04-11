# written by: James Kim
# date: Apr 10, 2023
# References
#   [1] "U-Net: Convolutional Networks for Biomedical Image Segmentation"

import torch
from torch import nn
from collections import OrderedDict
from .unet_model import UNet
from .utils import train_network, test_network
from .MEOIdataset import MEOIdataset

def main():
    # use our dataset and defined transformations
    dataset = MEOIdataset('data')
    dataset_test = MEOIdataset("data")

    #split the data dataset in train and test set
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size = 2, shuffle = True)

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size = 1, shuffle = False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    n_channels = 4
    n_classes = 2
    
    # get the model
    model = UNet(n_channels=n_channels, n_classes=n_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma=0.1)

    # train it for 5 epochs
    n_epochs = 5
    log_interval = 10
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs+1)]

    for epoch in range(n_epochs):
        # train for one epoch, printing every 10 iterations
        train_network(epoch, model, optimizer, log_interval, train_loader, train_losses, train_counter, batch_size_train)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        test_network(model, test_loader, test_losses)



if __name__ == "__main__":
    main()