import torch
import inference
import dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import load_model, save_model, load_inference_model
from config import name, LR, momentum, freeze_rate, gamma, step_size, weight_decay, log_dir, RESUME, path_checkpoint,\
    model_path
from train import train_classifier, train_classifier_resume


def main(train=True):

    # data
    train_loader, valid_loader, test_loader = dataset.get_dataset()

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    model = load_model(name)
    if model.classifier is not None:
        fc_params_id = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * freeze_rate},  # 0
            {'params': model.classifier.parameters(), 'lr': LR}], momentum=momentum, weight_decay=weight_decay)
    elif model.fc is not None:
        fc_params_id = list(map(id, model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * freeze_rate},  # 0
            {'params': model.fc.parameters(), 'lr': LR}], momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if train:
        writer = SummaryWriter(log_dir=log_dir, comment='test_tensorboard')
        if not RESUME:
            train_classifier(model, 100, train_loader, valid_loader, optimizer, criterion, scheduler, writer=writer)
        else:
            train_classifier_resume(model, optimizer, path_checkpoint, 100, train_loader, valid_loader, criterion,
                                scheduler, writer=writer)
        save_model(model, name, save_state_dic=True)
    else:
        state_dict = torch.load(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_model = load_inference_model(name)
        inference_model.load_state_dict(state_dict)
        inference_model.to(device)
        inference_model.eval()
        pred_list1 = []
        pred_list2 = []
        pred_list3 = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images, _ = data
                images = images.to(device)

                # tensor to vector
                outputs = inference_model(images)
                _, pred_top3 = torch.topk(outputs, k=3, dim=-1)

                pred_list1.append(pred_top3.data[:, 0].cpu().numpy().reshape((1, -1)).tolist())
                pred_list2.append(pred_top3.data[:, 1].cpu().numpy().reshape((1, -1)).tolist())
                pred_list3.append(pred_top3.data[:, 2].cpu().numpy().reshape((1, -1)).tolist())
        lists = [pred_list1, pred_list2, pred_list3]
        inference.vector2label(lists)


if __name__ == '__main__':
    # Training
    TRAIN = True
    main(TRAIN)

    # Inference
    TRAIN = False
    main(TRAIN)
