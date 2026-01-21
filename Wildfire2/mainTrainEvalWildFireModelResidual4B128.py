from wildfire_model import WildFireNet2DV3L_3x3_Residual as WildFireNet
import os
import argparse
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import numpy as np

from datasetAugmentationWildFire4B  import WildFireDataset
from tqdm import tqdm


'''

python mainTrainEvalWildFireModelResidual12B128.py  --mode train -b 256  --epochs 5000
 
'''

SEGNET_LOCAL = True


parse = argparse.ArgumentParser()
parse.add_argument(
    '--mode', choices=['train', 'eval', 'valid', 'test', 'eval_diff'])
parse.add_argument('--batch_size', '-b', type=int, default=16)
parse.add_argument('--resume', type=str, default="None")
parse.add_argument('--start_epoch', type=int, default=1)
parse.add_argument('--epochs', type=int, default=5000)
parse.add_argument('--min_loss', type=float, default=float('inf'))
parse.add_argument('--max_f1score', type=float, default=0.)
parse.add_argument('--dims', type=str, default="(32,64)")
parse.add_argument('--lr', type=float, default=0.01)

args = parse.parse_args()


data = "Wildfire2"


data_root = "/home/liese2/SPRI_AI_project/" + data

dir_img = os.path.join(data_root, 'Images')
dir_mask = os.path.join(data_root, 'SegmentationClass')
train_path = os.path.join(data_root, "ImageSets/Segmentation/train.txt")
val_path = os.path.join(data_root, "ImageSets/Segmentation/valid.txt")
nb_classe = 2


#print("directories ")
#print("dir img path ",dir_img)
#print("dir mask ",dir_mask)
#print("train path ",train_path)
#print("val path ",train_path)

  
SIZE = 128
# SIZE= 64


STEPVALID = 1 # 2  # 2 #5

NORMALIZE = True

#data += "_2023_"

if NORMALIZE:
    data += "_N_"


data += "_"+str(SIZE)

data += "_"


niv_aug=1
# niv_aug=2
augmentation=True




#data += args.dims.replace("(", "_").replace(")", "_").replace(",", "-")

data += "_AUG_"+ str(niv_aug)

print("Model file:", data)

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device ", device)

class_loss_weight = torch.Tensor([1.0, 1.3]).to(device)

min_loss = args.min_loss
f1score_max = args.max_f1score

print("Create modele ..")

dim = eval(args.dims)

model = WildFireNet(4, nb_classe, dims=dim) #numero de canales


if args.resume != "None":

    file = args.resume
    print("Load module ...", file)

    model.load_state_dict(torch.load(file))
    model.eval()

 
model.to(device)


print("Optimizer ...")
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


criterion = nn.CrossEntropyLoss(weight=class_loss_weight)



def train(epoch, epochs, train_dataloader, n_train):
    global min_loss

    model.train()

    start_time = time.time()
    mean_loss = 0
    idx = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

        for batch in train_dataloader:
            imgs = torch.autograd.Variable(batch['image'])
            label = torch.autograd.Variable(batch['mask'])

            img = imgs.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = model(img)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()

            idx += 1
            pbar.update(imgs.shape[0])

    mean_loss /= idx
    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'epoch {epoch} loss: {mean_loss}, elapse time: {elapse_time}')

    if mean_loss < min_loss:
        print(f'in epoch {epoch}, loss decline')
        min_loss = mean_loss
        torch.save(model.state_dict(), data+'_best.pth')
        with open(data+"min_loss.txt", "a") as f:
            f.write(f'epoch {epoch} min_loss {min_loss} \n')

    with open("trace_epoch_segnet.txt", "a") as f:
        f.write(
            f'epoch {epoch} loss: {mean_loss}, elapse time: {elapse_time}\n')


def valid(epoch, epochs, val_dataloader, n_val, save=True):
    global f1score_max

    model.eval()

    TP, FP, FN,TN = 0, 0, 0,0
    all=0

    with tqdm(total=n_val, desc=f'validation', unit='img') as pbar:
        idx = 0

        for batch in val_dataloader:
            imgs = torch.autograd.Variable(batch['image'])
            label = torch.autograd.Variable(batch['mask'])
         
            
            img = imgs.to(device)
            label = label.to(device)
          
 
            if SEGNET_LOCAL:
                output = model(img )
            else:
                _, output = model(img )

            label = label.cpu()
            _, predict = torch.max(output, dim=1)
            pred = predict.cpu().numpy()

            for b in range(imgs.size()[0]):
                pred_t=np.equal(pred[b,:,:],1)
                label_t=np.equal(label[b,:,:],1)

                tp_a=np.logical_and(pred_t,label_t)

                tp=np.count_nonzero(tp_a)

                TP+=tp

                pred_c1=np.count_nonzero(pred_t)
                fp= pred_c1 - tp
                FP+=fp

                label_c1=np.count_nonzero(label_t)
                fn=label_c1-tp
                FN+=fn

                   
                all_=imgs.size()[2]*imgs.size()[3]
                TN+=all_-(tp+fp+fn)
                all+=all_
  

            pbar.update(imgs.shape[0])
            idx += 1

    recall, precision, f1score = 0., 0., 0.
    if (TP + FN) != 0 and (TP + FP) != 0:
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * recall * precision) / (recall + precision)
    acc= (TP + TN)/all       
 
    stri=   f'{epoch} R {recall:.6} Acc {acc:.6f} P {precision:.6} F1score {f1score:.6} {data} \n'   
    print('\n --- > valid epoch ',stri)
 
   
    if save:
        with open("f1score_trace_128.txt", "a") as f:
            f.write(stri)
            f.write("\n")
        if f1score > f1score_max:

            f1score_max = f1score
            print("F1score augment")
            print(data)
            torch.save(model.state_dict(), data+'_valid_best_3264.pth')
            stri+=data
            with open("f1score_max_trace_128.txt", "a") as f:
                f.write(stri)
                f.write("\n")

def train_valid(epochs):

    batch_size = args.batch_size
    print("WildFireTrainDataset ... ", train_path, dir_img, dir_mask)
    train_dataset = WildFireDataset(list_file=train_path,
                                 img_dir=dir_img,
                                 mask_dir=dir_mask,
                                 size=SIZE,
                                 augmentation=augmentation,
                                 niv=niv_aug,
                                 normalize=NORMALIZE

                                 )
    print("DataLoader ... ")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  #shuffle=False,
                                  num_workers=8)  # 4 A MODIFIER

    print("WildFireValDataset ... ", val_path, dir_img, dir_mask)
    val_dataset = WildFireDataset(list_file=val_path,
                               img_dir=dir_img,

                               mask_dir=dir_mask,
                               normalize=NORMALIZE,
                               size=SIZE)

    n_val = len(val_dataset)
    print("DataLoader ... ")

    if batch_size > 4:
        batch_size = batch_size//4

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                #                                  batch_size=1,
                                shuffle=False,
                                num_workers=8)  # 4 A MODIFIEIER

    n_train = len(train_dataset)
    # valid(0,epochs,val_dataloader,n_val,save=False)
    for i in range(args.start_epoch, epochs+1):

        train(i, epochs, train_dataloader, n_train)

        # if  (n_train > 25000  and  i% 5 ==0)  or (n_train < 25000 and i% 10 == 0)  :
        if i % STEPVALID == 0:
            valid(i, epochs, val_dataloader, n_val)


def valid_():
    epochs = 0

    batch_size = args.batch_size

    print("WildFireValDataset ... ", val_path, dir_img, dir_mask)
    val_dataset = WildFireDataset(list_file=val_path,
                               img_dir=dir_img,
                               mask_dir=dir_mask,
                               #mask_cote=dir_cote,
                               normalize=NORMALIZE,
                               size=SIZE)

    n_val = len(val_dataset)
    print("DataLoader ... ")

    if batch_size > 4:
        batch_size = batch_size//4

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                #                                  batch_size=1,
                                shuffle=False,
                                num_workers=8)  # 4 A MODIFIEIER

    valid(0, epochs, val_dataloader, n_val, save=False)



  

def eval():
    batch_size = args.batch_size

    if args.resume == "None":

        file = "segnet_WildFire_best.pth"
        print("load", file)
        model.load_state_dict(torch.load(file))

    model.eval()

    print("WildFireValDataset ... ", val_path, dir_img, dir_mask)
    val_dataset = WildFireDataset(list_file=val_path,
                               img_dir=dir_img,
                               mask_dir=dir_mask,
                               size=SIZE,
                               normalize=NORMALIZE)

    n_val = len(val_dataset)
    print("DataLoader ... ")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8)  # 4 A MODIFIEIER

    mean_loss = 0

    with tqdm(total=n_val, desc=f'iteration', unit='img') as pbar:
        start_time = time.time()
        idx = 0

        for batch in val_dataloader:
            imgs = torch.autograd.Variable(batch['image'])
            label = torch.autograd.Variable(batch['mask'])

            img = imgs.to(device)
            labels = label.to(device)
            nb = imgs.shape[0]

            if SEGNET_LOCAL:
                output = model(img)
            else:
                _, output = model(img)

            output = model(img)

            i = 0
            for label, pred in zip(labels, output):

                pred = output[i].float()

                loss = F.cross_entropy(pred.unsqueeze(
                    dim=0), label.unsqueeze(dim=0)).item()
                mean_loss += loss

                if idx % 10 == 0:
                    print('Loss/train', loss, 'global loss',
                          mean_loss / (idx+1), 'step', idx, flush=True)
                i += 1

            loss = 0
            pbar.update(nb)
            idx += 1

        mean_loss /= idx
    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'loss: {mean_loss}, elapse time: {elapse_time}')


def test():

    if args.resume == "None":

        file = "segnet_WildFire_best.pth"
        print("load", file)
        model.load_state_dict(torch.load(file))

    model.eval()


    print("WildFireValDataset ... ", val_path, dir_img, dir_mask)
    val_dataset = WildFireDataset(list_file=val_path,
                               img_dir=dir_img,
                               mask_dir=dir_mask,
                               size=SIZE,
                               normalize=NORMALIZE)

    n_val = len(val_dataset)
    print("DataLoader ... ")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)  # 4 A MODIFIEIER



    with tqdm(total=n_val, desc=f'iteration', unit='img') as pbar:
        idx = 0

        for batch in val_dataloader:
            imgs = torch.autograd.Variable(batch['image'])
            label = torch.autograd.Variable(batch['mask'])

            img = imgs.to(device)
            label = label.to(device)

            label = label.squeeze()

            if SEGNET_LOCAL:
                output = model(img)
            else:
                _, output = model(img)

            label = label.cpu()
            _, predict = torch.max(output, dim=1)
            pred = predict.cpu().numpy()
            name = val_dataset.get_name(idx)
            pred = pred.squeeze()
            plt.imsave(f'./test/{name}_mask', label)

            plt.imsave(f'./test/{name}_predict', pred)

            pbar.update(imgs.shape[0])
            idx += 1


def eval_diff_img(t1, t2):

    sum = 0
    for c in range(nb_classe):
        n1 = np.equal(t1, c)
        n2 = np.equal(t2, c)

        idt = np.logical_and(n1, n2)

        nb = np.count_nonzero(idt)

        sum += nb

    return sum/(t1.shape[0]*t1.shape[0])


def eval_diff():

    if args.resume == "None":

        file = "segnet_WildFire_best.pth"
        print("load", file)
        model.load_state_dict(torch.load(file))

    model.eval()

    val_path = train_path  # A MODIFIER EVENTUELLEMENT

    print("WildFireValDataset ... ", val_path, dir_img, dir_mask)
    val_dataset = WildFireDataset(list_file=val_path,
                               img_dir=dir_img,
                               mask_dir=dir_mask,
                               size=SIZE)

    n_val = len(val_dataset)
    print("DataLoader ... ")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)  # 4 A MODIFIEIER

    total = 0

    with tqdm(total=n_val, desc=f'iteration', unit='img') as pbar:
        idx = 0

        for batch in val_dataloader:
            imgs = torch.autograd.Variable(batch['image'])
            label = torch.autograd.Variable(batch['mask'])

            img = imgs.to(device)
            label = label.to(device)

            label = label.squeeze()

            if SEGNET_LOCAL:
                output = model(img)
            else:
                _, output = model(img)

            label = label.cpu()
            _, predict = torch.max(output, dim=1)
            pred = predict.cpu().numpy()
            name = val_dataset.get_name(idx)
            pred = pred.squeeze()

            for cl in range(nb_classe):
                print(np.sum(pred == cl), end="\t")
            print()

            diff = eval_diff_img(label, pred)

            total += diff

            if idx % 100 == 0:
                print("gene ", diff, total / (idx + 1))


            pbar.update(imgs.shape[0])
            idx += 1
    print("diff ganeral= ", total/n_val)


if __name__ == '__main__':
    if args.mode == 'train':
        train_valid(args.epochs)
    elif args.mode == 'eval':
        eval()
    elif args.mode == 'valid':
        valid_()
    elif args.mode == 'test':
        test()
    elif args.mode == 'eval_diff':
        eval_diff()
