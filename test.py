import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import model

batch_size_test = 1000

# Load data from mnist
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# Load our model and model state from training
network = model.Net()
network_state_dict = torch.load('results/model.pth')
network.load_state_dict(network_state_dict)

# Disable gradient calculation to reduce memory usage
with torch.no_grad():
  output = network(example_data)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
  plt.show()
print(fig)