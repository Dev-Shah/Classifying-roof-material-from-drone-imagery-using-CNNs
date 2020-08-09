import torch
from torch import nn, optim
from torchvision import models
from torch.optim import lr_scheduler
from torch.autograd import Variable
#for corrupted images
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

import time
import copy
from collections import OrderedDict
from load_data import load_train_val_data
from cosine_annealing import cosine_annealing

def basic_training(dir_,model_name, optimizer_name, num_epochs, learning_rate, weight_decay, step_size, gamma, device):

    trainloader, valloader = load_train_val_data(dir_)
    try:
        model = getattr(models, model_name)(pretrained = True)
    except:
        print("Model %s doesn't exist" % (model_name))
        print("Select one of the below listed models")
        print([item for item in dir(models) if item[0]!='_' ])

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(OrderedDict([
                  ('fcc', nn.Linear(2048, 5)),
                  #('relu', nn.ReLU()),
                  #('fc', nn.Linear(1000, 5)),
                  ('output', nn.LogSoftmax(dim=1))
                  ]))

    if device == 'cuda':
        model.cuda()

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    best_loss = 1
    train_losses = []
    val_losses = []

    dataloaders={'train':trainloader,'val':valloader}
    dataset_sizes = {'train': len(trainloader.dataset), 'val':len(valloader.dataset)}

    criterion = nn.NLLLoss()

    try:
        optimizer = getattr(optim,optimizer_name)(model.fc.parameters(), lr=learning_rate, weight_decay = weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    except:
        print("Optimizer %s doesn't exist" % (optimizer_name))
        print("Select one of the below listed optimizers")
        print([item for item in dir(optim) if item[0]!='_' ])




    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            if phase == 'train':
                train_losses.append(epoch_loss)

            else:
                val_losses.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))


    train_losses_name = "/models/tr_loss_%s_optimizer_%s_lr_%.2f_wd_%.2f_epochs_%d" % (model_name,optimizer_name, learning_rate, weight_decay, num_epochs)
    with open(train_losses_name,"wb") as f:
        pickle.dump(train_losses,f)

    val_losses_name = "/models/val_loss_%s_optimizer_%s_lr_%.2f_wd_%.2f_epochs_%d" % (model_name,optimizer_name, learning_rate, weight_decay, num_epochs)
    with open(val_losses_name,"wb") as f:
        pickle.dump(val_losses,f)

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_name = "/models/%s_optimizer_%s_lr_%.2f_wd_%.2f_epochs_%d" % (model_name,optimizer_name, learning_rate, weight_decay, num_epochs)
    print("Saving at %s",(save_name))
    torch.save(model.state_dict(),("/models/%s_optimizer_%s_lr_%.2f_wd_%.2f_epochs_%d" % (model_name,optimizer_name, learning_rate, weight_decay, num_epochs)))

    return model

def snapshot_ensembling_training(dir_,model_name, optimizer_name, num_epochs, num_cycles, lr_max, device, weighted_average = False):



    if weighted_average:
        print('Weighted Average')
        trainloader, valloader, valloader_2 = load_train_val_data(dir_,weighted_average = True)
    else:
        trainloader, valloader = load_train_val_data(dir_)

    try:
        model = getattr(models, model_name)(pretrained = True)
    except:
        print("Model %s doesn't exist in the pretrained models list" % (model_name))
        print("Select one of the below listed models")
        print([item for item in dir(models) if item[0]!='_' ])

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(OrderedDict([
                  ('fcc', nn.Linear(2048, 5)),
                  #('relu', nn.ReLU()),
                  #('fc', nn.Linear(1000, 5)),
                  ('output', nn.LogSoftmax(dim=1))
                  ]))


    if device == 'cuda':
        model.cuda()

    cosine_series = cosine_annealing(num_epochs, num_cycles, lr_max)

    ratio = num_epochs/num_cycles

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    best_loss = 1
    train_losses = []
    val_losses = []

    dataloaders={'train':trainloader,'val':valloader}
    dataset_sizes = {'train': len(trainloader.dataset), 'val':len(valloader.dataset)}

    criterion = nn.NLLLoss()

    optimizer = getattr(optim,optimizer_name)(model.fc.parameters(), lr=lr_max)

    models_list = []

    for epoch in range(num_epochs):
        try:
            optimizer = getattr(optim,optimizer_name)(model.fc.parameters(), lr=cosine_series[epoch])
        except:
            print("Optimizer %s doesn't exist" % (optimizer_name))
            print("Select one of the below listed optimizers")
            print([item for item in dir(optim) if item[0]!='_' ])

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('Learning Rate')
        print(cosine_series[epoch])
        since_for_loop=time.time()
        if (epoch+1)%ratio==0:
            torch.save(model.state_dict(),("/models/%s_snapshot_optimizer_%s_lr_%.2f_cycle_%d" % (model_name,optimizer_name, lr_max, ((epoch+1)/ratio))))
            models_list.append(copy.deepcopy(model))
            print('Cycle')
            print((epoch+1)/ratio)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:


            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)

            else:
                val_losses.append(epoch_loss)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(),'resnet_model_aug_true_')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    train_losses_name = "/models/tr_loss_%s_snapshot_optimizer_%s_lr_%.2f_cycle_%d" % (model_name,optimizer_name, lr_max, ((epoch+1)/ratio))
    with open(train_losses_name,"wb") as f:
        pickle.dump(train_losses,f)

    val_losses_name = "/models/val_loss_%s_snapshot_optimizer_%s_lr_%.2f_cycle_%d" % (model_name,optimizer_name, lr_max, ((epoch+1)/ratio))
    with open(val_losses_name,"wb") as f:
        pickle.dump(val_losses,f)


    # load best model weights
    model.load_state_dict(best_model_wts)
    return models_list
