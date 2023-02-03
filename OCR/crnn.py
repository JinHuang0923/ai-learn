import torch
import torch.nn as nn
import numpy as np
# 在IIIT-5K中，我们的label中的字符有A-Z，0-9，还有别忘了空字符。一共37个。
class CRNN(nn.Module):
    """
    CRNN模型

    Args:
        in_channels (int): 输入的通道数, 如果是灰度图则为1, 如果没有灰度化则为3
        out_channels (int): 输出的通道数（类别数），即样本里共有多少种字符
    """
    def __init__(self, in_channels, out_channels):
        super(CRNN, self).__init__()
        self.in_channels = in_channels
        hidden_size = 256
        # CNN 结构与参数
        self.cnn_struct = ((64, ), (128, ), (256, 256), (512, 512), (512, ))
        self.cnn_paras = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (2, 1, 0))
        # 池化层结构
        self.pool_struct = ((2, 2), (2, 2), (2, 1), (2, 1), None)
        # 是否加入批归一化层
        self.batchnorm = (False, False, False, True, False)
        self.cnn = self._get_cnn_layers()
        # RNN 两层双向LSTM。pytorch中LSTM的输出通道数为hidden_size * num_directions,这里因为是双向的，所以num_directions为2
        self.rnn1 = nn.LSTM(self.cnn_struct[-1][-1], hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True)
        # 最后一层全连接
        self.fc = nn.Linear(hidden_size*2, out_channels)
        # 初始化参数，不是很重要
        self._initialize_weights()

    def forward(self, x):           # input: height=32, width>=100
        x = self.cnn(x)             # batch, channel=512, height=1, width>=24
        x = x.squeeze(2)            # batch, channel=512, width>=24
        x = x.permute(2, 0, 1)      # width>=24, batch, channel=512
        x = self.rnn1(x)[0]         # length=width>=24, batch, channel=256*2
        x = self.rnn2(x)[0]         # length=width>=24, batch, channel=256*2
        l, b, h = x.size()
        x = x.view(l*b, h)          # length*batch, hidden_size*2
        x = self.fc(x)              # length*batch, output_size
        x = x.view(l, b, -1)        # length>=24, batch, output_size
        return x

    # 构建CNN层
    def _get_cnn_layers(self):
        cnn_layers = []
        in_channels = self.in_channels
        for i in range(len(self.cnn_struct)):
            for out_channels in self.cnn_struct[i]:
                cnn_layers.append(
                    nn.Conv2d(in_channels, out_channels, *(self.cnn_paras[i])))
                if self.batchnorm[i]:
                    cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if (self.pool_struct[i]):
                cnn_layers.append(nn.MaxPool2d(self.pool_struct[i]))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
# 我们用一个类来实现编码解码。要注意的是，因为我们所用的CTCLoss库的实现中，默认将空字符编码为0(可以通过参数修改)，所以我们要为其余字符设置从1开始的编码。
class LabelTransformer(object):
    """
    字符编码解码器

    Args:
        letters (str): 所有的字符组成的字符串
    """
    def __init__(self, letters):
        self.encode_map = {letter: idx+1 for idx, letter in enumerate(letters)}
        self.decode_map = ' ' + letters

    def encode(self, text):
        if isinstance(text, str):
            length = [len(text)]
            result = [self.encode_map[letter] for letter in text]
        else:
            length = []
            result = []
            for word in text:
                length.append(len(word))
                result.extend([self.encode_map[letter] for letter in word])
        return torch.IntTensor(result), torch.IntTensor(length)

    def decode(self, text_code):
        result = []
        for code in text_code:
            word = []
            for i in range(len(code)):
                if code[i] != 0 and (i == 0 or code[i] != code[i-1]):
                    word.append(self.decode_map[code[i]])
            result.append(''.join(word))
        return result