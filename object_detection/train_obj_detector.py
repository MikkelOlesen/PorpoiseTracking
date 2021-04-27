import torch
import torchvision
import mytransforms as T
import torch.utils.data as data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_utils.engine import train_one_epoch, evaluate
from pytorch_utils import utils




from create_detection_dataset import porpoise_dataset

IMG_RESIZE = 800
BATCH_SIZE = 8
NUM_WORKERS = 2
DATA_PATH = "resnet_object_detector"
TRAIN_SPLIT = 0.1 

TRANSFORM_TRAIN = T.Compose([
    T.ToTensor(),
    T.Resize(IMG_RESIZE),
    #T.RandomVerticalFlip(0.5),
    T.RandomHorizontalFlip(0.5),
    T.RandomColor(0.6,0.4,0.5,0.2),
    T.AddRandomNoise(0.05,0.5),
    T.ShowImg(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

TRANSFORM_VAL= T.Compose([
    T.ToTensor(),
    T.Resize(IMG_RESIZE),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    # train on the GPU or on the CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Create datasets with the right transforms
    train_dataset = porpoise_dataset(DATA_PATH, TRANSFORM_TRAIN)
    val_dataset = porpoise_dataset(DATA_PATH, TRANSFORM_VAL)

    # Spiltting the dataset train and validation 90/10
    split_pct = int(len(train_dataset)*TRAIN_SPLIT)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-split_pct])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-split_pct:])

    dataloader_train = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=utils.collate_fn)
    dataloader_val = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=utils.collate_fn)

    # Using pretrained resnet50 model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, max_size=1000)

    # replace the classifier with a clasifier for only porpoise and bg
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Send to GPU
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dataloader_val, device=device)

    torch.save(model, DATA_PATH + "/model")

if __name__ == "__main__":
    main()