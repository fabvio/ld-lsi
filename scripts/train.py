import torch
import torchvision
import os
from glob import glob
from PIL import Image
from utils import AverageMeter, iou, debug_val_example



class CrossEntropyLoss2d(torch.nn.Module):
    #NLLLoss2d is negative log-likelihood loss
    #it returns the semantic segmentation cross-entropy loss2d
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        outputs, _ = outputs
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets) 

def train_one_epoch(model, criterion, optimizer, data_loader, debug_data_loader, device, ntrain_batches):
    top1 = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for data in data_loader:
        cnt += 1
        model.to(device)
        print('.',end='')
        image, target = data['image'].to(device), data['label'].squeeze(1)
        target = torch.round(target*255) #converting to range 0-255
        target = target.type(torch.int64).to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1 = iou(output, target)
        top1.update(acc1, image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt%10 == 0:
            #debug val example checks and prints a random debug example every
            #10 batches.
            #it uses cpu, so might be slow. 
            debug_val_example(model, debug_data_loader)
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f}'
                        .format(top1=top1))
            
        if cnt >= ntrain_batches:
            return 

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f}'
                .format(top1=top1))
    return

class berkely_driving_dataset(torch.utils.data.Dataset):
    def __init__(self, path, type='train', transform=None, color = True):
    # dataloader for bdd100k segmentation dataset
    # path should contain the address to bdd100k folder
    # it generally has the following diretory structure
        """
        - bdd100k
          - drivable_maps
            - color_labels
            - labels
          - images
            - 100k
            - 10k
        """
        # type can either be 'train' or 'val'
        self.path = path
        self.type = type
        self.transform = transform
        self.imgs = glob(os.path.join(self.path, 'images/100k/' + self.type + '/*.jpg'))
        if color is True:
            self.labels = [os.path.join(self.path, 'drivable_maps/color_labels/' + self.type,\
                        x.split('/')[-1][:-4] + '_drivable_color.png') for x in self.imgs]
        else:
            self.labels = [os.path.join(self.path, 'drivable_maps/labels/' + self.type,\
                                     x.split('/')[-1][:-4]+'_drivable_id.png') for x in self.imgs]
        self.length = len(self.imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imgs[idx]
        image = Image.open(img_name)
        label = Image.open(self.labels[idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return {'image':image, 'label':label}


if __name__=='__main__':
    """
    Here is an example of how to use these functions for training
    This script is designed to train on berkely driving dataset. Therefore, the 
    PATH_TO_BERKELY_DATASET variable points to the root of that dataset. You might
    have to edit it.
    """

    #DEFINING SOME IMPORTANT VARIABLES
    PATH_TO_BERKELY_DATASET = 'bdd100k'

    #loading libraries
    from res.models import erfnet_road

    #loading models
    roadnet = erfnet_road.Net()
    if torch.cuda.is_available():
        device = torch.device("cuda")  
    else:
        device = torch.device("cpu")  

    #loading weights
    model_w = torch.load('res/weights/weights_erfnet_road.pth')
    new_mw = {}
    for k,w in model_w.items():
        new_mw[k[7:]] = w
    roadnet.load_state_dict(new_mw)

    roadnet.to(device)
    roadnet.eval();


    # Making Dataloaders
    bdd_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((360, 640)),
        #torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])

    bdd_train = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms,  type='train', color = False)
    bdd_val = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms,  type='val', color = False)

    sampler_train = torch.utils.data.SequentialSampler(bdd_train)
    sampler_val = torch.utils.data.RandomSampler(bdd_val)

    dl_train = torch.utils.data.DataLoader(
        bdd_train, batch_size=5,
        sampler=sampler_train)

    # the valiation only works with a batchsize of 1
    dl_val = torch.utils.data.DataLoader(
        bdd_val, batch_size=1,
        sampler=sampler_val)

    #defining losses
    criterion = CrossEntropyLoss2d()
    optimizer = torch.optim.Adam(roadnet.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    #training an epoch for 100 batches
    train_one_epoch(roadnet, criterion, optimizer, dl_train, dl_val, device, 100)