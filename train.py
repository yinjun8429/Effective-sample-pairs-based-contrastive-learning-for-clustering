import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import pandas
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from multi_crop import MultiCropDataset
from modules.nn_memory_bank import NNMemoryBankModule
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        ###
        z_i = memory_bank(z_i,update=False) #NNMemoryBankModule: return to NN
        z_j = memory_bank(z_j,update=False)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(   
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch

# def train():
#     loss_epoch = 0
#     #feature_space = get_score(model, device, train_dataloader, test_dataloader)
#     for step, ((x_i, x_j), _) in enumerate(data_loader):
#         optimizer.zero_grad()
#         x_i = x_i.to('cuda')
#         x_j = x_j.to('cuda')
#         z_i, z_j, c_i, c_j = model(x_i, x_j)
#         z_i = z_i-center
#         z_i = z_j-center
#         ###
#         z_i = memory_bank(z_i,update=False)
#         z_j = memory_bank(z_j,update=False)
#         loss_instance = criterion_instance(z_i, z_j)
#         loss_cluster = criterion_cluster(c_i, c_j)
#         loss = loss_instance + loss_cluster
#         loss.backward()
#         optimizer.step()
#         if step % 50 == 0:
#             print(   
#                 f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
#         loss_epoch += loss.item()
#     return loss_epoch

# def get_score(model, device, train_dataloader, test_dataloader):
#     train_feature_space = []
#     with torch.no_grad():
#         for (imgs, _) in tqdm(train_dataloader, desc='Train set feature extracting'):
#             imgs = imgs.to(device)
#             features = model(imgs)
#             features = normalize( NNCLRProjectonHead(512,512,128)(features), dim=1)
#             train_feature_space.append(features)
#         train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
#     return train_feature_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
           
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = MultiCropDataset(
            data_path="datasets/tiny-imagenet/tiny-imagenet-200/train"
        )
        # dataset = torchvision.datasets.ImageFolder(
        #     root='datasets/tiny-imagenet/tiny-imagenet-200/train',
        #     transform=transform.Transforms(s=0.5, size=args.image_size),
        # )
        class_num = 200
    else:
        raise NotImplementedError
        
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num).to(device)
    model = nn.DataParallel(model)
    model = model.to('cuda')

    memory_bank = NNMemoryBankModule(size=4095)
    memory_bank.to(device)

    
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    #using Contrastive_Loss
    criterion_instance = Contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device,args.tau_plus,args.beta,args.estimator).to(loss_device)
    criterion_cluster = Contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    
    # criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(loss_device)
    # criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 50 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
