from vision.network import CRNN
import torch
import torchinfo
from util.data_loader import RegDataSet
from torchvision.transforms import transforms

def eval_model():
    net.eval()
    data_loader = DataLoader(valSet, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_iter = iter(data_loader)
    i = 0
    loss_avg = 0.0

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        images, labels, target_lengths, input_lengths = next(val_iter)
        i += 1
        preds = net(images)
        cost = ctc_loss(log_probs=preds, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)
        loss_avg += cost
    print("val loss: {}".format(loss_avg / max_iter))
    net.train()
characters = "-0123456789"

net = CRNN(len(characters))
device = torch.device("cuda")
net.load_state_dict(torch.load("./weights/Final.pth", map_location=torch.device(device)))
# net.eval()
print(net)
torchinfo.summary(net)
transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

valSet = RegDataSet(dataset_root="./test", anno_txt_path="annotation_val.txt", lexicon_path="lexicon.txt",
                    target_size=(200, 32), characters=characters, transform=transform)