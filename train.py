import torch
import torchvision
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from data.utils import getData
from sklearn.metrics import accuracy_score
from torch import nn
# from thop import profile
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from CompactNet import CompactNet

def train(model, train_loader, test_loader, args, device='cpu'):
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=0.0001,
    )
    model = model.train()
    for e in range(args.epoch):
        running_loss = 0
        correct = 0
        total = 0
        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_f(output, y)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
        running_loss /= len(train_loader)
        print('Training loss: ', running_loss)
        print('Training acc: ', correct / total)
        writer.add_scalar('Training Acc', correct / total, e + 1)
        writer.add_scalar('Training Loss', running_loss, e + 1)
        logging.info('Training Acc: {}'.format(correct / total))
        scheduler.step()
        print('Current lr: ', scheduler.get_last_lr())


        acc, loss = validate(model, test_loader, loss_f, device)
        print(f'The {e + 1} epoch validation accuracy is: {acc}')
        print(f'The {e + 1} epoch validation loss is: {loss}')
        writer.add_scalar('Validation Acc', acc, e + 1)
        writer.add_scalar('Validation Loss', loss, e + 1)
        logging.info('Validation Acc: {}'.format(acc))
        model_path = os.path.join(args.root, f"{args.network}_acc{round(acc, 4)}.pth")
        # torch.save(model.state_dict(), model_path)

def validate(network, loader, loss_f, device):
    network.eval()
    predictions = []
    labels = []
    loss = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            outputs = network(data)
            loss += loss_f(outputs, target)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            labels.extend(target.tolist())
    network.train()
    loss /= len(loader)
    accuracy = accuracy_score(labels, predictions)
    return accuracy, loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--root', type=str, default=r'E:\Direction')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--base_lr', type=int, default=1e-3)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--network', type=str, default='CompactNet',
                        help='networks')
    args = parser.parse_args()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    writer = SummaryWriter(log_dir='runs/{}'.format(args.network))
    trainset, valset = getData(args.root)
    train_loader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batchsize, shuffle=True)
    logging.info(f'''Starting training:
                         Batch size:      {args.batchsize}
                         Learning rate:   {args.base_lr}
                         Dataset size:    {len(train_loader) + len(val_loader)}
                         Training size:   {len(train_loader)}
                         Validation size: {len(val_loader)}''')

    model = CompactNet(num_classes=3)

    model.to(device)
    train(model, train_loader, val_loader, args, device)