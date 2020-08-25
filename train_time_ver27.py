import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np

from model_skip50 import*

if_check_point = False

Time = 3

epoch_num = 2
#batch_size = 15
batch_size = 15
chanel_num = 3
#image_file_num = 1640
image_file_num = 100
light_kind_num = 5

block_num = 10
block_set_num = image_file_num*block_num
block_size = 64
train_data_num = chanel_num*block_set_num*light_kind_num
check_num = train_data_num


view_point_row_num = 5
view_point_num = view_point_row_num*view_point_row_num

cnt = 0
loss_sum = 0.0


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight.data)


if os.path.isdir("./Dst") == False:
    os.mkdir("./Dst")
if os.path.isdir("./Dst/Model") == False:
    os.mkdir("./Dst/Model")
if os.path.isdir("./Dst/Model/ver27") == False:
    os.mkdir("./Dst/Model/ver27")


# Load train datas
load_light_field = np.zeros(
    (block_set_num*chanel_num, view_point_num, block_size, block_size), np.uint8)
start_index_R = 0
start_index_G = image_file_num * block_num
start_index_B = image_file_num * block_num * 2

aug_data = np.zeros((batch_size, view_point_num,
                     block_size, block_size), np.float32)


for n in range(image_file_num):
    if(n % 10 == 0):
        print("\r load %04d/%04d" % (n, image_file_num), end="")

    fname = "/src_data/%04d.png" % (n)
    img_load = Image.open(fname)
    img_load = np.asarray(img_load)

    for b in range(block_num):
        for t in range(view_point_num):
            R_img = img_load[block_size * b: block_size * b +
                             block_size, block_size * t: block_size * t + block_size, 0]
            G_img = img_load[block_size * b: block_size * b +
                             block_size, block_size * t: block_size * t + block_size, 1]
            B_img = img_load[block_size * b: block_size * b +
                             block_size, block_size * t: block_size * t + block_size, 2]

            load_light_field[n*block_num+b +
                             start_index_R, t, :, :] = R_img
            load_light_field[n*block_num+b +
                             start_index_G, t, :, :] = G_img
            load_light_field[n*block_num+b +
                             start_index_B, t, :, :] = B_img

"""
# Load test datas
test_lf = []
for t in range(view_point_row_num):
    for s in range(view_point_row_num):
        fname = "/home/matsuura/Dataset/full_data/drabunny/5x5/input_Cam%03d.png" % (
            t * view_point_row_num + s)
        img_load = Image.open(fname)
        img_load = np.asarray(img_load)
        img_load = (img_load / 255.0).astype(np.float32)
        test_lf.append(img_load.copy())

size_h = test_lf[0].shape[0]
size_w = test_lf[0].shape[1]

test_light_field = np.zeros( (3, view_point_num, size_h, size_w), np.float32)

for u in range(view_point_num):
    test_light_field[0, u, :, :] = test_lf[u][:, :, 0]
    test_light_field[1, u, :, :] = test_lf[u][:, :, 1]
    test_light_field[2, u, :, :] = test_lf[u][:, :, 2]

lumi_test_light_field = test_light_field * 0.5
"""


# Model initaition
layer_cnn = LayerCNN()
# layer_cnn.apply(init_weight)
layer_cnn = layer_cnn.cuda()
layer_cnn_optimizer = optim.Adam(layer_cnn.parameters(), lr=0.0001)
scheduler = StepLR(layer_cnn_optimizer, step_size=15, gamma=0.5)
# layer_cnn_optimizer = optim.SGD(layer_cnn.parameters(), lr=0.0001)
# layer_cnn_optimizer = optim.Adagrad(layer_cnn.parameters(), lr=0.0001)


start_epoch = 1

if if_check_point:
    check_point = torch.load("./Dst/Model/ver27/checkpoint.pt")
    layer_cnn.load_state_dict(check_point["state_dict"])
    layer_cnn_optimizer.load_state_dict(check_point["optimizer"])
    start_epoch = check_point["epoch"]


# Train Loop
for epoch in range(start_epoch, epoch_num + 1):
    layer_cnn.train()
    print("epoch : " + str(epoch))
    # for param_group in layer_cnn_optimizer.param_groups:
    #print("lr : " + str(param_group["lr"]))
    sff = np.random.permutation(train_data_num)

    for n in range(0, train_data_num, batch_size):
        # Data augmentation by intensity levels (ここを変えると良い？)
        for j in range(batch_size):
            aug_data[j, :, :, :] = ((1.0 - (sff[n + j] % light_kind_num)/10) *
                                    load_light_field[int(sff[n + j] / light_kind_num)] / 255).astype(np.float32)
        gray_light_field = torch.Tensor(aug_data)
        gray_light_field = gray_light_field.cuda()
        # gray_light_field = torch.tensor(gray_light_field)

        # Train models
        layer_cnn_optimizer.zero_grad()
        gray_layer = layer_cnn(gray_light_field)
        gray_layer = torch.unsqueeze(gray_layer, 2)

        gray_reconstruct_light_field = gray_layer[:,
                                                  0, :, 0: block_size - 4, 0: block_size - 4]

        loss = 0.0

        for t in range(view_point_row_num):
            for p in range(view_point_row_num):
                gray_reconstruct_light_field = torch.cat([gray_reconstruct_light_field, (gray_layer[:, 0, :, t: block_size - 4 + t, p: block_size - 4 + p]
                                                                                         * gray_layer[:, 1, :, 2: block_size - 2, 2: block_size - 2, ] * gray_layer[:, 2, :, 4 - t: block_size - t, 4 - p: block_size - p] + gray_layer[:, 3, :, t: block_size - 4 + t, p: block_size - 4 + p]
                                                                                         * gray_layer[:, 4, :, 2: block_size - 2, 2: block_size - 2, ] * gray_layer[:, 5, :, 4 - t: block_size - t, 4 - p: block_size - p] + gray_layer[:, 6, :, t: block_size - 4 + t, p: block_size - 4 + p]
                                                                                         * gray_layer[:, 7, :, 2: block_size - 2, 2: block_size - 2, ] * gray_layer[:, 8, :, 4 - t: block_size - t, 4 - p: block_size - p])/Time], dim=1)
        """
        for j in range(batch_size):
            gray_reconstruct_light_field[j, :, :, :][gray_reconstruct_light_field[j, :, :, :] > (
                0.1 + (sff[n + j] % light_kind_num) / 10)] = 0.1 + (sff[n + j] % light_kind_num) / 10
        """

        loss = F.mse_loss(gray_reconstruct_light_field[:, 1:view_point_num+1,
                                                       :, :, ], gray_light_field[:, :, 2: block_size - 2, 2: block_size - 2])
        loss.backward()
        layer_cnn_optimizer.step()
        scheduler.step(epoch)
        print("\repoch:%04d/%04d, data:%06d/%06d, mse:%.6f" %
              (epoch, epoch_num, n, train_data_num, 10 * loss), end="")
        loss_sum += loss.detach()
        cnt += 1

        # Save masks and calculate average psnr
        if n % check_num == 0:
            print()
            print("@@@    average psnr:%.6f    @@@" %
                  (10*torch.log10(1.0 / (loss_sum / cnt))))
            f_psnr = open("./Dst/average_psnr.txt", "a")
            f_psnr.write("epoch:%04d/%04d, data:%06d/%06d, psnr:%.6f\n" %
                         (epoch, epoch_num, n, train_data_num, 10*torch.log10(1.0 / (loss_sum / cnt))))
            f_psnr.close()

            cnt = 0
            loss_sum = 0.0

    # Save models every epoch
    torch.save(layer_cnn.state_dict(),
               "./Dst/Model/ver27/layer_cnn%04d.pt" % epoch)

    if epoch % 2 == 0:
        state = {"epoch": epoch, "state_dict": layer_cnn.state_dict(),
                 "optimizer": layer_cnn_optimizer.state_dict()}
        torch.save(state, "./Dst/Model/ver27/checkpoint.pt")

    """
    layer_cnn.eval()
    with torch.no_grad():

        gray_test_light_field = torch.Tensor(test_light_field)
        lumi_gray_test_light_field = torch.Tensor(lumi_test_light_field)
        gray_test_light_field = gray_test_light_field.cuda()
        lumi_gray_test_light_field = lumi_gray_test_light_field.cuda()

        gray_test_layer = layer_cnn(lumi_gray_test_light_field)

        gray_test_layer = torch.unsqueeze(gray_test_layer, 2)
        gray_test_reconstruct_light_field = gray_test_layer[:,
                                                            0, :, 0:size_h-4, 0:size_w-4]

        loss = 0.0

        for t in range(view_point_row_num):
            for p in range(view_point_row_num):
                gray_test_reconstruct_light_field = torch.cat([gray_test_reconstruct_light_field, (gray_test_layer[:, 0, :, t: size_h - 4 + t, p: size_w - 4 + p]
                                                                                                   * gray_test_layer[:, 1, :, 2: size_h - 2, 2: size_w - 2, ] * gray_test_layer[:, 2, :, 4 - t: size_h - t, 4 - p: size_w - p] + gray_test_layer[:, 3, :, t: size_h - 4 + t, p: size_w - 4 + p]
                                                                                                   * gray_test_layer[:, 4, :, 2: size_h - 2, 2: size_w - 2, ] * gray_test_layer[:, 5, :, 4 - t: size_h - t, 4 - p: size_w - p] + gray_test_layer[:, 6, :, t: size_h - 4 + t, p: size_w - 4 + p]
                                                                                                   * gray_test_layer[:, 7, :, 2: size_h - 2, 2: size_w - 2, ] * gray_test_layer[:, 8, :, 4 - t: size_h - t, 4 - p: size_w - p])/Time], dim=1)
        print()
        gray_test_reconstruct_light_field /= 0.5
        gray_test_reconstruct_light_field[gray_test_reconstruct_light_field > 1.0] = 1.0
        loss = F.mse_loss(gray_test_reconstruct_light_field[:, 1:view_point_num+1,
                                                            :, :, ], gray_test_light_field[:, :, 2: size_h - 2, 2: size_w - 2])
        print("test PSNR : %.6f" % (10 * torch.log10(1.0 / loss)))
    """
