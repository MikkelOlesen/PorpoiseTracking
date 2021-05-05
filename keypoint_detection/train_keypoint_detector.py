import torch
import torchvision
import torch.utils.data as data
from torch import nn

from create_keypoint_dataset import porpoise_keypoint_dataset
import myKeypointTransforms as T

IMG_RESIZE = 224
BATCH_SIZE = 2
NUM_WORKERS = 1
SAVE_MODEL_DIR = "keypoint_detection"
DATA_PATH = "porpoise_keypoint_data"
TRAIN_SPLIT = 0.1
num_epochs = 10

TRANSFORM_TRAIN = T.Compose([
    T.ToTensor(),
    T.RandomColor(0.6,0.4,0.5,0.2),
    T.AddRandomNoise(0.05,0.5),
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

def train(dataloader_train, dataloader_val, model, criterion, optimizer, epochs, device):

    #Save losses for plot
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
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
            
            # backward pass compute gradient
            loss.backward()

            # parameter update
            optimizer.step()

            # update running training loss
            train_loss += loss.item()*batch['image'].size(0)


        model.eval() # prep model for validation

        for batch in dataloader_val:
            # forward pass (get predictions)
            output = model(batch['image'].to(device))

            # flatten keypoints to be 1x8 instead of 2x4
            keypoints = torch.flatten(batch['keypoints'], start_dim=1)

            # calculate the loss (error)
            loss = criterion(output, batch['keypoints'].to(device))
            
            # update running validation loss 
            valid_loss += loss.item()*batch['image'].size(0)

        # print training/validation statistics 
        # calculate average Root Mean Square loss over an epoch
        train_loss = np.sqrt(train_loss/len(dataloader_train.sampler.indices))
        valid_loss = np.sqrt(valid_loss/len(dataloader_val.sampler.indices))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
              .format(epoch+1, train_loss, valid_loss))

    return train_losses, valid_losses 

def main():
    # train on the GPU or on the CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Create datasets with the right transforms
    train_dataset = porpoise_keypoint_dataset(DATA_PATH, TRANSFORM_TRAIN)
    val_dataset = porpoise_keypoint_dataset(DATA_PATH, TRANSFORM_VAL)

    # Spiltting the dataset train and validation 90/10
    split_pct = int(len(train_dataset)*TRAIN_SPLIT)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-split_pct])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-split_pct:])

    dataloader_train = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=NUM_WORKERS, pin_memory=True)
    dataloader_val = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=NUM_WORKERS, pin_memory=True)

    # Using pretrained resnet50 model
    model = torchvision.models.resnet50(pretrained=True)

    # replace the last layer to fit all keypoints
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4*2)


    # Send to GPU
    model.to(device)

    # construct an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    train_losses, valid_losses = train(dataloader_train, dataloader_val, model, criterion, optimizer, 10, device)

    torch.save(model, SAVE_MODEL_DIR + "/model")

if __name__ == "__main__":
    main()