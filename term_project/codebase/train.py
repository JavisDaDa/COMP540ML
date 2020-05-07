import torch
import numpy as np
from config import MAX_EPOCH, checkpoint_interval, val_interval, checkpoint_path


def train_classifier(model, print_every, train_loader, valid_loader, optimizer, criterion, scheduler, writer=None):
    train_curve = list()
    valid_curve = list()
    model.to('cuda')
    iter_count = 0
    start_epoch = -1
    for e in range(start_epoch + 1, MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_loader):
            iter_count += 1
            # forward
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            output = model.forward(images)

            # backward
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # calculate loss and training infomation
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()

            # print
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % print_every == 0:
                loss_mean = loss_mean / print_every
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    e, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

            # save to event file
            writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
            writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)

        scheduler.step()
        # for every epoch, record weight and gradient
        for name, param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, e)
            writer.add_histogram(name + '_data', param, e)

        # save checkpoint
        if (e + 1) % checkpoint_interval == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": e,
                "iter_count": iter_count
            }
            path_checkpoint = checkpoint_path + f'checkpoint_{e}_epoch.pkl'
            torch.save(checkpoint, path_checkpoint)

        # validation
        if (e + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            top1 = 0.
            top3 = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    images, labels = data
                    images, labels = images.to('cuda'), labels.to('cuda')

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    _, maxk = torch.topk(outputs, k=3, dim=-1)
                    total_val += labels.size(0)
                    labels = labels.view(-1, 1)
                    # correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
                    top1 += (labels == maxk[:, 0:1]).sum().cpu().item()
                    top3 += (labels == maxk).sum().cpu().item()
                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)
                valid_curve.append(loss.item())
                print(
                    "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Top1_Acc:{:.2%} Top3_Acc:{:.2%}".format(
                        e, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, top1 / total_val, top3 / total_val))
                # save to event file
                writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iter_count)
                writer.add_scalars("Top1Accuracy", {"Valid": top1 / total_val}, iter_count)
                writer.add_scalars("Top3Accuracy", {"Valid": top3 / total_val}, iter_count)
            model.train()


def train_classifier_resume(model, optimizer, path_checkpoint, print_every, train_loader, valid_loader, criterion, scheduler, writer=None):
    checkpoint = torch.load(path_checkpoint)
    model.to('cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    iter_count = checkpoint['iter_count']
    train_curve = list()
    valid_curve = list()
    for e in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_loader):
            iter_count += 1
            # forward
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            output = model.forward(images)

            # backward
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # calculate loss and training infomation
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()

            # print
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % print_every == 0:
                loss_mean = loss_mean / print_every
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    e, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.
            # save to event file
            writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
            writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)

        # change learning rate
        scheduler.step()

        # for every epoch, record weight and gradient
        for name, param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, e)
            writer.add_histogram(name + '_data', param, e)

        if (e + 1) % checkpoint_interval == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": e,
                "iter_count": iter_count
            }
            path_checkpoint = checkpoint_path + f'checkpoint_{e}_epoch.pkl'
            torch.save(checkpoint, path_checkpoint)

        # validation
        if (e + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            top1 = 0.
            top3 = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    images, labels = data
                    images, labels = images.to('cuda'), labels.to('cuda')

                    outputs = model.forward(images)
                    loss = criterion(outputs, labels)

                    _, maxk = torch.topk(outputs, k=3, dim=-1)
                    total_val += labels.size(0)
                    labels = labels.view(-1, 1)
                    # correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
                    top1 += (labels == maxk[:, 0:1]).sum().cpu().item()
                    top3 += (labels == maxk).sum().cpu().item()
                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)
                valid_curve.append(loss.item())
                print(
                    "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Top1_Acc:{:.2%} Top3_Acc:{:.2%}".format(
                        e, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, top1 / total_val, top3 / total_val))
                # save to event file
                writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iter_count)
                writer.add_scalars("Top1Accuracy", {"Valid": top1 / total_val}, iter_count)
                writer.add_scalars("Top3Accuracy", {"Valid": top3 / total_val}, iter_count)
            model.train()