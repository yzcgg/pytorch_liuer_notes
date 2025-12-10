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
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5,padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        outputs = [branch_pool, branch1x1, branch5x5, branch3x3]
        # (batch-size, C, W, H) cat according to C, so it is dim=1
        return torch.cat(outputs, dim=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88,20,kernel_size=5)

        self.mp = torch.nn.MaxPool2d(2)

        self.icpt1 = InceptionA(10)
        self.icpt2 = InceptionA(20)

        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.icpt1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.icpt2(x)
        x = x.view(in_size, -1)
        return self.fc(x)

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# train function
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('Epoch:', epoch, '\tLoss:', running_loss / len(train_loader))

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            outputs = model(test_inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    print('Accuracy : %d %% [%d/%d]' % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()