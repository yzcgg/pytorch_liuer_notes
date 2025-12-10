import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load train data
train_dataset = datasets.MNIST(
    root='./dataset',
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load test data
test_dataset = datasets.MNIST(
    root='./dataset',
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x) # will use cross entropy, do not activate

model = Net()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# train function
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('Epoch:', epoch, '\tLoss:', running_loss / len(train_loader))

# test function
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            test_inputs, test_labels = data
            outputs = model(test_inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    print('Accuracy : %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()