import torch
import torch.nn as nn
from config import N_CLASSES
from torchvision import models


def load_model(name : str):
    if name.startswith('resnext101'):
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext101_32x8d', pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('resnet152'):
        model = models.resnet152(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('resnet101'):
        model = models.resnet101(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('wide_resnet101'):
        model = models.wide_resnet101_2(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('densenet161'):
        model = models.densenet161(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('densenet169'):
        model = models.densenet169(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('densenet201'):
        model = models.densenet201(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, N_CLASSES)
        return model
    else:
        raise(ValueError('Select another model'))


def load_inference_model(name : str):
    if name.startswith('resnext101'):
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext101_32x8d', pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('resnet152'):
        model = models.resnet152(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('resnet101'):
        model = models.resnet101(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('wide_resnet101'):
        model = models.wide_resnet101_2(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('densenet161'):
        model = models.densenet161(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('densenet169'):
        model = models.densenet169(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, N_CLASSES)
        return model
    elif name.startswith('densenet201'):
        model = models.densenet201(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, N_CLASSES)
        return model
    else:
        raise(ValueError('Select another model'))


def save_model(model, name, save_state_dic=False):
    model_path = f'./drive/My Drive/COMP540/{name}.pkl'
    if save_state_dic:
        path_state_dict = f'./drive/My Drive/COMP540/{name}_state_dict.pkl'
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, path_state_dict)
    torch.save(model, model_path)
