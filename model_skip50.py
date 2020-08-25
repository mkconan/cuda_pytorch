import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerCNN(nn.Module):
    def __init__(self):
        super(LayerCNN, self).__init__()
        self.cn01 = nn.Conv2d(25, 64, 3, padding=1)
        self.cn02 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn03 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn04 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn05 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn06 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn07 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn08 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn09 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn10 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn11 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn12 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn13 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn14 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn15 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn16 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn17 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn18 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn19 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn20 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn21 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn22 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn23 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn24 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn25 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn26 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn27 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn28 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn29 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn30 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn31 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn32 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn33 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn34 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn35 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn36 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn37 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn38 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn39 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn40 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn41 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn42 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn43 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn44 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn45 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn46 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn47 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn48 = nn.Conv2d(64, 64, 3, padding=1)
        self.cn49 = nn.Conv2d(64, 64, 3, padding=1)

        self.cnL = nn.Conv2d(64, 9, 3, padding=1)

    def forward(self, x):
        f = F.relu(self.cn01(x))
        residual = f
        f = F.relu(self.cn02(f))
        f = F.relu(self.cn03(f) + residual)
        residual = f
        f = F.relu(self.cn04(f))
        f = F.relu(self.cn05(f) + residual)
        residual = f
        f = F.relu(self.cn06(f))
        f = F.relu(self.cn07(f) + residual)
        residual = f
        f = F.relu(self.cn08(f))
        f = F.relu(self.cn09(f) + residual)
        residual = f
        f = F.relu(self.cn10(f))
        f = F.relu(self.cn11(f) + residual)
        residual = f
        f = F.relu(self.cn12(f))
        f = F.relu(self.cn13(f) + residual)
        residual = f
        f = F.relu(self.cn14(f))
        f = F.relu(self.cn15(f) + residual)
        residual = f
        f = F.relu(self.cn16(f))
        f = F.relu(self.cn17(f) + residual)
        residual = f
        f = F.relu(self.cn18(f))
        f = F.relu(self.cn19(f) + residual)
        residual = f
        f = F.relu(self.cn20(f))
        f = F.relu(self.cn21(f) + residual)
        residual = f
        f = F.relu(self.cn22(f))
        f = F.relu(self.cn23(f) + residual)
        residual = f
        f = F.relu(self.cn24(f))
        f = F.relu(self.cn25(f) + residual)
        residual = f
        f = F.relu(self.cn26(f))
        f = F.relu(self.cn27(f) + residual)
        residual = f
        f = F.relu(self.cn28(f))
        f = F.relu(self.cn29(f) + residual)
        residual = f
        f = F.relu(self.cn30(f))
        f = F.relu(self.cn31(f) + residual)
        residual = f
        f = F.relu(self.cn32(f))
        f = F.relu(self.cn33(f) + residual)
        residual = f
        f = F.relu(self.cn34(f))
        f = F.relu(self.cn35(f) + residual)
        residual = f
        f = F.relu(self.cn36(f))
        f = F.relu(self.cn37(f) + residual)
        residual = f
        f = F.relu(self.cn38(f))
        f = F.relu(self.cn39(f) + residual)
        residual = f
        f = F.relu(self.cn40(f))
        f = F.relu(self.cn41(f) + residual)
        residual = f
        f = F.relu(self.cn42(f))
        f = F.relu(self.cn43(f) + residual)
        residual = f
        f = F.relu(self.cn44(f))
        f = F.relu(self.cn45(f) + residual)
        residual = f
        f = F.relu(self.cn46(f))
        f = F.relu(self.cn47(f) + residual)
        residual = f
        f = F.relu(self.cn48(f))
        f = F.relu(self.cn49(f) + residual)
        f = self.cnL(f)
        hard_sigmoid = MyHaraSigmoid.apply
        return hard_sigmoid(f)


class MyHaraSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = (0.2 * input + 0.5).clamp(min=0.0, max=1.0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        input, = ctx.saved_tensors
        grad_input *= 0.2
        grad_input[input < -2.5] = 0
        grad_input[input > 2.5] = 0

        return grad_input
