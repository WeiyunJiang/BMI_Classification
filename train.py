import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import random
import time
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import utils
import torch
import torch.nn as nn

from args import breast_arg
from tqdm import tqdm
from models import VGG_16, Alex_Net, Efficient_Net, SE_Net
from dataio import Breast_Dataset, Modality_Dataset
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix

def validation(model, model_dir, val_data_loader, epoch, total_steps, 
               best_val_acc, args, criterion):
    with torch.no_grad():
        total_acc = []
        total_loss = []
        model.eval()
        summaries_dir = os.path.join(model_dir, 'val_summaries')
        utils.cond_mkdir(summaries_dir)
    
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)
        
        writer = SummaryWriter(summaries_dir) # initialize tensorboard

        for step, batch in tqdm(enumerate(val_data_loader)):  
            
            image, label, feature = batch['image'], batch['label'], batch['feature']
            image = image.to(device)
            label = label.to(device)
            
            
            pred = model(image.float())
            pred = pred.squeeze(-1)
            label = label.type(torch.LongTensor)
            
            label = label.to(device)
            loss = criterion(pred, label)
            # pred[pred > 0.5] = 1
            # pred[pred <= 0.5] = 0
            total_loss.append(loss.clone().detach().cpu().numpy())
            correct_results_sum = (torch.argmax(pred, dim=-1) == label).sum().float()
        
            acc = correct_results_sum/pred.shape[0]
            total_acc.append(acc.clone().detach().cpu().numpy())
        # record validation loss and accuracy
        writer.add_scalar("step_val_loss", np.mean(total_loss), total_steps)
        writer.add_scalar("step_val_acc", np.mean(total_acc), total_steps)
        

        tqdm.write("Val Loss: %.4f, acc: %.4f" 
                   % (np.mean(total_loss), np.mean(total_acc)))
        # store the model pth of the best validation accuracy
        if np.mean(total_acc) > best_val_acc:
            best_val_acc = np.mean(total_acc)
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_best_val.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'best_acc_epoch.txt'),
                       np.array([best_val_acc, epoch]))
def train_model(model, model_dir, train_data_loader, val_data_loader,
                args, summary_fn=None, device=None):
    ''' Traning script

    '''
    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    writer = SummaryWriter(summaries_dir) # initial tensorboard
    criterion = nn.CrossEntropyLoss() # define loss function
    model.train(True) 
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_steps = 0
    best_val_acc = 0
    with tqdm(total=len(train_data_loader) * args.epochs) as pbar:
        for epoch in range(args.epochs):
            print("Epoch {}/{}".format(epoch, args.epochs))
            print('-' * 10)
            epoch_train_losses = []
            epoch_train_acc = []
            if not (epoch+1) % args.epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                # save the pth periodically
            for step, batch in enumerate(train_data_loader):
                start_time = time.time()
                optimizer.zero_grad()
                image, label = batch['image'], batch['label']
                image = image.to(device)
                label = label.to(device)
                
                pred = model(image.float())
                pred = pred.squeeze()
                label = label.type(torch.LongTensor)
                label = label.squeeze()
                label = label.to(device)
                loss = criterion(pred, label)
                loss.backward()
                # pred[pred > 0.5] = 1 # clip the output to 0 and 1
                # pred[pred <= 0.5] = 0
                
                correct_results_sum = (torch.argmax(pred, dim=-1) == label).sum().float()
                acc = correct_results_sum/label.shape[0]
                epoch_train_acc.append(acc.clone().detach().cpu().numpy())
                epoch_train_losses.append(loss.clone().detach().cpu().numpy())
                clip_grad_norm_(model.parameters(), 0.1)  
                optimizer.step()
                pbar.update(1)
                tqdm.write("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f, iteration time %0.6f sec" 
                % (epoch, args.epochs, step, len(train_data_loader), loss, acc, time.time() - start_time)) 
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_current.pth'))
                # save the current path
                writer.add_scalar("step_train_loss", loss, total_steps)
                for param_group in optimizer.param_groups:
                    writer.add_scalar("epoch_train_lr", param_group['lr'], total_steps)         
                total_steps += 1
            # record the training accuracy and loss
            writer.add_scalar("epoch_train_loss", np.mean(epoch_train_losses), epoch)
            writer.add_scalar("epoch_train_acc", np.mean(epoch_train_acc), epoch)  
            ## validation
            validation(model, model_dir, val_data_loader, epoch, total_steps, best_val_acc, args, criterion)


if __name__ == '__main__': 
    args = breast_arg()
    # Set random seed
    seed = 40
    print(f'Using random seed {seed}')
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    root_path = os.path.join(args.logging_root, args.exp_name)
    
    
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(device) 
    num_features = 2
    if args.model == 'vgg':
        model = VGG_16(num_features, args.cat_feat)
    elif args.model == 'alexnet':
        model = Alex_Net(num_features, args.cat_feat)
    elif args.model == 'effnet':
        model = Efficient_Net(num_features, args.cat_feat)
    elif args.model == 'senet':
        model = SE_Net(num_features, args.cat_feat)
    else:
        raise NotImplementedError('Not implemented for name={args.name}')
    
    
    
    model.to(device) 
    total_n_params = utils.count_parameters(model)
    print(f'Total number of parameters of {args.model}: {total_n_params}')
    
    train_dataset = Modality_Dataset(split='train', data_aug=args.data_aug)
    num_train_imgs = 781
    num_val_imgs = len(train_dataset) - num_train_imgs
    img_dataset_train, img_dataset_val = random_split(train_dataset, \
                                                         [num_train_imgs, 
                                                          num_val_imgs]
                                                          )
    test_dataset = Modality_Dataset(split='test', data_aug=False)
    
    
    train_data_loader = DataLoader(img_dataset_train, batch_size=args.bt, shuffle=True)
    val_data_loader = DataLoader(img_dataset_val, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    train_model(model, root_path, train_data_loader, val_data_loader,
                args, summary_fn=None, device=device)





