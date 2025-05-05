import netron
import torch.onnx
from torch.autograd import Variable
from model import get_gcnet

myNet = get_gcnet()
# 调整输入尺寸为兼容的奇数值（例如65x257）
left = torch.randn(1, 3, 375, 1242)
right = torch.randn(1, 3, 375, 1242)
modelData = "./demo.onnx"
# 传递两个输入参数
torch.onnx.export(myNet, (left, right), modelData)
netron.start(modelData)