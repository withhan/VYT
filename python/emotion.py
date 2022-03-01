import torch
import numpy as np
import torch.nn as nn
import collections

from PIL import Image
from torchvision import transforms

import time
import pickle

import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

data_buffer = collections.deque()

def send_data(data):
    global data_buffer
    data_buffer.append(data)
    if len(data_buffer) > 10:
        data_buffer.popleft()
    msg = 'C#' + str(collections.Counter(data_buffer).most_common()[0][0])
    sock.sendto(msg.encode(), ('192.168.0.8', 8003))
model_weight = collections.OrderedDict()

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.PReLU(),
            nn.ZeroPad2d(2),
            nn.MaxPool2d(kernel_size=5, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.PReLU(),
            nn.ZeroPad2d(padding=1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.PReLU()
        )

        self.layer5 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.PReLU(),
            nn.ZeroPad2d(1),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Linear(3200, 1024)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.prelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.prelu(x)
        x = self.dropout(x)

        y = self.fc3(x)
        y = self.log_softmax(y)

        return y

transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

def createmodel(path):
    global model_weight
    layer_name = ['layer1.0.weight', 'layer1.0.bias', 'layer1.1.weight', 'layer2.1.weight', 'layer2.1.bias',
                  'layer2.2.weight', 'layer3.0.weight', 'layer3.0.bias', 'layer3.1.weight', 'layer4.1.weight',
                  'layer4.1.bias', 'layer4.2.weight', 'layer5.1.weight', 'layer5.1.bias', 'layer5.2.weight',
                  'fc1.weight', 'fc1.bias', 'prelu.weight', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
    for n, i in enumerate(layer_name):
        buffer = np.load(
            path + str(n) + '.npy',
            allow_pickle=False)
        model_weight[i] = torch.from_numpy(buffer)

class_labels_kor = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
class_labels = ['Happy', 'Surprised', 'Angry', 'Unrest', 'Wound', 'Sad', 'Neutral']
class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2, '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}

device = torch.device('cpu')
model = EmotionNet()

createmodel("model/MyEmotionModelConvert/")
model.load_state_dict(model_weight)
best_trained_model = model.to(device)

best_trained_model.eval()

print("시동 프로세스 완료")
image_save_path = './image/'
last_time = 0
sense_time = 1/20
onece = True

while True:
    if time.time() - last_time < sense_time:
        continue
    try:
        with open(image_save_path + '/fdata.pickle', "rb") as fr:
            frame = pickle.load(fr)
    except FileNotFoundError as e:
        print("경로가 설정이 잘못되어있습니다")
        print("프로그램을 종료합니다")
        break
    except FileExistsError:
        continue
    except EOFError:
        continue
    last_time = time.time()
    targetimg = Image.fromarray(frame)
    target = transform_test(targetimg)
    targetreshape = np.reshape(target, (1, 1, 48, 48))
    inputdata = targetreshape.float().to(device)
    result = best_trained_model(inputdata)
    maxindex = result.max(1, keepdim=True)[1].numpy().flatten()[0]

    # send data
    send_data(maxindex)
    print(maxindex)
