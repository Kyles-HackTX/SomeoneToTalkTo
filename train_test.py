import torch
import torch.nn as nn
import torch.nn.functional as F


def train():
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        ep_run_loss = 0.0

        for i, data in enumerate(training_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = nn.CrossEntropyLoss()(torch.log(outputs), labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            ep_run_loss += loss.item()
            if i % 5 == 0:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

        print(f"Epoch {epoch} loss: {ep_run_loss / i}")
    print('Finished Training')

    return


def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # calculate outputs by running images through the network
            outputs = model(inputs.float())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print(total, labels, labels.size(0))
            print(predicted)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    print('Finished Testing')

    return
