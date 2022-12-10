import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

# 通道顺序
# [batch, channel, height, width]

# 图像预处理函数
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 保存位置，是不是训练集，是否需要下载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

# 这个东西是用来加载数据的，一次性加载全部数据内存不够，因此每次加载36张图片，之后每次都打乱，然后线程的话，在windows下面都要是0
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.__next__()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
# 损失函数使用交叉熵函数，包含了SoftMax以及NLLLoss
loss_function = nn.CrossEntropyLoss()
# 优化器，对所有参数都训练，然后学习率是多少
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data
        # 清除历史损失
        optimizer.zero_grad()
        outputs = net(inputs)
        # 看看本次的输出和预期值的损失
        loss = loss_function(outputs, labels)
        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 500步打印一次，里面的内容不计算误差梯度
        if(step % 500 == 499):
            with torch.no_grad():
                outputs = net(test_image)  # 这里输出的维度是[batch, 10]
                # 直接要最大的就行，最大的就是预测的
                predict_y = torch.max(outputs, dim=1)[1]
                # 对的是1，不对的是0
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                # 这里的loss是每500算了一次，所以记得除一个500
                print('[%d, %5d train_loss: %.3f test_accuracy: %3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))

print('Finished Training')

save_path = 'model/LeNet.pth'
torch.save(net.state_dict(), save_path)
