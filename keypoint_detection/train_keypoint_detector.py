import torch
import torchvision
import torch.utils.data as data
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt

from create_keypoint_dataset import porpoise_keypoint_dataset
import myKeypointTransforms as T

IMG_RESIZE = 224
BATCH_SIZE = 30
NUM_WORKERS = 2
SAVE_MODEL_DIR = "keypoint_detection"
DATA_PATH = "porpoise_keypoint_data"
TRAIN_SPLIT = 0.1
num_epochs = 1000



TRANSFORM_TRAIN = T.Compose([
    T.ToTensor(),
    T.RandomColor(0.4,0.2,0.3,0.1),
    T.AddRandomNoise(0.02,0.5),
    T.RandomFlip(),
    T.Square_Pad(),
    T.Resize(IMG_RESIZE),
    #T.ShowImg(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

TRANSFORM_VAL= T.Compose([
    T.ToTensor(),
    T.Square_Pad(),
    T.Resize(IMG_RESIZE),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train(dataloader_train, dataloader_val, model, criterion,criterion_val, optimizer, epochs, device, scheduler):

    #Save losses for plot
    train_losses = []
    valid_losses = []

    valid_loss_min = 10

    for epoch in range(epochs):
        start_time = time.time()

        # init losses 
        train_loss = 0.0
        valid_loss = 0.0

        model.train() # prep model for training

        for batch in dataloader_train:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass (get predictions)
            output = model(batch['image'].to(device))

            # flatten keypoints to be 1x8 instead of 2x4
            keypoints = torch.flatten(batch['keypoints'], start_dim=1)

            # calculate the loss (error)
            loss = criterion(output, keypoints.to(device))
            loss_mean_square = criterion_val(output, keypoints.to(device))
            
            # backward pass compute gradient
            loss.backward()

            # parameter update
            optimizer.step()

            # update running training loss
            train_loss += loss_mean_square.item()*batch['image'].size(0)        
        
        model.eval() # prep model for validation
        with torch.no_grad():
            for batch in dataloader_val:
                # forward pass (get predictions)
                output = model(batch['image'].to(device))

                # flatten keypoints to be 1x8 instead of 2x4
                keypoints = torch.flatten(batch['keypoints'], start_dim=1)
                
                # calculate the loss (error)
                loss = criterion(output, keypoints.to(device))
                loss_mean_square = criterion_val(output, keypoints.to(device))
                
                # update running validation loss 
                valid_loss += loss_mean_square.item()*batch['image'].size(0)
        
        if(scheduler != None):
            scheduler.step(train_loss)
        # print training/validation statistics 
        # calculate average Root Mean Square loss over an epoch
        train_loss = np.sqrt(train_loss/len(dataloader_train.dataset))
        valid_loss = np.sqrt(valid_loss/len(dataloader_val.dataset))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss <= valid_loss_min:
            print("Saving model with:", valid_loss)
            torch.save(model, SAVE_MODEL_DIR + "/model")
            valid_loss_min = valid_loss

        end_time = time.time()
        time_taken = end_time-start_time

        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} Time: {:.2f} \tLR: {:.7f}'
              .format(epoch+1, train_loss, valid_loss, time_taken, optimizer.param_groups[0]['lr']))

    return train_losses, valid_losses 

def main():
    # train on the GPU or on the CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')

    # Create datasets with the right transforms
    train_dataset = porpoise_keypoint_dataset(DATA_PATH, TRANSFORM_TRAIN)
    val_dataset = porpoise_keypoint_dataset(DATA_PATH, TRANSFORM_VAL)

    # Spiltting the dataset train and validation 90/10
    split_pct = int(len(train_dataset)*TRAIN_SPLIT)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-split_pct])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-split_pct:])

    dataloader_train = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    dataloader_val = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=NUM_WORKERS, pin_memory=True)

    # Using pretrained resnet50 model
    model = torchvision.models.resnet34(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = True 

    # replace the last layer to fit all keypoints
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4*2)
    #model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 4*2))

    # Send to GPU
    model.to(device)

    # construct an optimizer
    criterion = nn.L1Loss()
    criterion_val = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.2,min_lr=0.0000001)
    scheduler = None

    train_losses, valid_losses = train(dataloader_train, dataloader_val, model, criterion,criterion_val, optimizer, num_epochs, device, scheduler)

    
    plt.plot(train_losses, color='b', label="Train")
    plt.plot(valid_losses, color='r', label="Validation")
    plt.xlabel("# of eproc")
    plt.ylabel("RMSE Loss")
    plt.ylim(0,50)
    plt.legend()
    plt.savefig(SAVE_MODEL_DIR + '/figures/lr_0_0001_R34_L1_LOSS_1000_new.png')
    plt.show()

if __name__ == "__main__":
    main()
