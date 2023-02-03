import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs['D'])

    def forward(self, x):
        x = self.features(x)
        return x


def make_layers(cfg, batch_norm=False):
    # 为了使用预训练的VGG权重, VGG backbone参照pytorch的VGG构造实现, 不然加载不了权重, 按照论文，第三层和第四层池化层核大小核步长改为（1， 2）


    layers = []
    in_channels = 3
    i = 0
    for v in cfg:
        if v == 'M':
            if i not in [9, 13]:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))]

        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
}


class BidirectionalLSTM(nn.Module):

    def __init__(self, inp, nHidden, oup):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(inp, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, oup)

    def forward(self, x):
        out, _ = self.rnn(x)
        T, b, h = out.size()
        t_rec = out.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, characters_classes, hidden=256, pretrain=True):
        super(CRNN, self).__init__()
        self.characters_class = characters_classes
        # Vgg特征提取
        self.body = VGG()
        # 将VGG stage5-1 卷积单独拿出来, 改了卷积核无法加载预训练参数
        self.stage5 = nn.Conv2d(512, 512, kernel_size=(3, 2), padding=(1, 0))
        self.hidden = hidden
        self.rnn = nn.Sequential(BidirectionalLSTM(512, self.hidden, self.hidden),
                                 BidirectionalLSTM(self.hidden, self.hidden, self.characters_class))

        self.pretrain = pretrain
        if self.pretrain:
            import torchvision.models.vgg as vgg
            pre_net = vgg.vgg16(pretrained=True)
            pretrained_dict = pre_net.state_dict()
            model_dict = self.body.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.body.load_state_dict(model_dict)

            for param in self.body.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.body(x)
        x = self.stage5(x)
        # 挤压掉高所在的维度
        x = x.squeeze(3)
        # 转换为LSTM所需格式
        x = x.permute(2, 0, 1).contiguous()
        x = self.rnn(x)
        x = F.log_softmax(x, dim=2)
        return x

def custom_collate_fn(batch, T=50):
        items = list(zip(*batch))
        items[0] = default_collate(items[0])
        labels = list(items[1])
        items[1] = []
        target_lengths = torch.zeros((len(batch,)), dtype=torch.int)
        input_lengths = torch.zeros(len(batch,), dtype=torch.int)
        for idx, label in enumerate(labels):
            # 记录每个图片对应的字符总数
            target_lengths[idx] = len(label)
            # 将batch内的label拼成一个list
            items[1].extend(label)
            # input_lengths 恒为 T
            input_lengths[idx] = T

        return items[0], torch.tensor(items[1]), target_lengths, input_lengths
def decode_out(str_index, characters):
    char_list = []
    for i in range(len(str_index)):
        if str_index[i] != 0 and (not (i > 0 and str_index[i - 1] == str_index[i])):
            char_list.append(characters[str_index[i]])
    return ''.join(char_list)

net_out = net(img)
_, preds = net_out.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)
lab2str = decode_out(preds, args.characters)
characters = "-0123456789"

ctc_loss = CTCLoss(blank=0, reduction='mean')

batch_iterator = iter(DataLoader(trainSet, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn))
images, labels, target_lengths, input_lengths = next(batch_iterator)
out = net(images)
loss = ctc_loss(log_probs=out, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)
