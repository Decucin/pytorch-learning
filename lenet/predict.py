import torch
from PIL import Image
from model import LeNet
import torchvision.transforms as transforms

# 因为不知道这个最后使用的时候输入是多少，所以先将其Resize为32 * 32的
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
# 这里把模型参数加载一下
net.load_state_dict(torch.load('model/LeNet.pth'))

# 可以开始预测了
im = Image.open('dog.jpg')
# 把它变为我们想要的输入
im = transform(im)
im = torch.unsqueeze(im, dim=0)

# 这里不需要学习
with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()

print(classes[int(predict)])
