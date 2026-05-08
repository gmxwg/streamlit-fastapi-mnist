import base64
import io

import torch
from fastapi import FastAPI, UploadFile
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

app = FastAPI(title = "手写数字识别")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt",map_location=torch.device('cpu'))) # .pt只有参数，要将参数字典保存在模型中，所以得先定义模型
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # 像素值从0-255变成0-1，从28 * 28 变成 1/3(通道数） * 28 * 28
    transforms.Normalize((0.1307,), (0.3081,))  # 从0 - 1变成-1 - 1，
])

@app.post("/get_result")
async def get_result_key(file : UploadFile): # UploadFile 就是 FastAPI 给你打包好的 “文件对象”，里面装着前端传过来的 3 样东西。

    image_bytes = await file.read() # .read() = 读取文件的真实内容，返回二进制字节。
    Img = Image.open(io.BytesIO(image_bytes)).convert("L") # 二进制字节 → 包装成虚拟文件 → 打开成图片 → 转成灰度图
    # ✅ 加上这一句：把灰色图 → 纯黑字 + 纯白背景
    Img = Img.point(lambda x: 0 if x < 180 else 255)

    # 图片转base64,让前端能显示
    buf = io.BytesIO()
    Img.save(buf, format="PNG")
    Img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    img = transform(Img).unsqueeze(0)
    with torch.no_grad():
        output = model(img).argmax(1).item()

    return {"result" : output, "Img_base64" : Img_base64}


