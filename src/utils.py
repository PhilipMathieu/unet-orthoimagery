# written by: James Kim
# date: Apr 10, 2023

import torch
import torch.nn.functional as F

# useful functions with a comment for each function
# trains the network
def train_network(epoch, network, optimizer, train_loader, log_interval, train_losses, train_counter, batch_size_train, save =False):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}({100.*batch_idx/len(train_loader):.0f}%)\tLoss: {loss.item():.6f}]')
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train)+ ((epoch-1) * len(train_loader.dataset)))
            
            ### E.Save the network to a file
            if save:
                torch.save(network.state_dict(), 'results/model.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer.pth')
    return

# test the network
def test_network(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%\n)'.format(
            test_loss, correct, len(test_loader.dataset),
            100.*correct / len(test_loader.dataset)))
    return