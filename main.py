import torch
import argparse
import numpy as np
from model import InvRat
from train import *
from data import get_loader, get_dataset
from torchvision import transforms
from utils import *
import sys


def get_args_parser():
    parser = argparse.ArgumentParser('VitInvRat', add_help=False)

    # Model parameters
    parser.add_argument('--k', default=80, type=int,
        help='Number of chosen tokens slots in Rationale Generator module')
    parser.add_argument('--num_classes', default=10, type=int,
        help='Number of classes for the dataset')
    parser.add_argument('--num_envs', default=2, type=int,
        help='Number of environments for the dataset')
    parser.add_argument('--model_type', default='vit_small_patch16_224_in21k', type=str,
        choices=['vit_small_patch16_224_in21k', 'vit_base_patch8_224_in21k',
                 'vit_tiny_patch16_224_in21k', 'vit_small_patch32_224_in21k',
                 'vit_small_patch8_224_dino', 'vit_base_patch16_224_in21k'],
        help="""Name of timm pretrain Vit. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--d_model', default=384, type=int, help="""Dimension of patch tokens corresponding to the model type""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory.""")
    parser.add_argument('--img_size', default=224, type=int, help="""Image Size""")
    parser.add_argument('--dim_classifier_head', default=128, type=int, help="""Dimension of the hidden layer of the classifier head""")
    parser.add_argument('--dim_generator_fc', default=64, type=int, help="""Dimension of the hidden layer of the generator fc""")
    
    # Loss parameters
    parser.add_argument('--var_lambda', default=10, type=float,
        help='Variance coefficient in invariance loss')
    parser.add_argument('--sparsity_percentage', default=0.2, type=float,
        help='Sparsity percentage in invariance loss')
    
    # Transformer parameters
    parser.add_argument('--num_layers', default=4, type=int,
        help='Number of layers in the predictor module')
    parser.add_argument('--dff', default=128, type=int,
        help='Dimension of feedforward in the transformer')
    parser.add_argument('--num_heads', default=1, type=int,
        help='Number of heads in the transformer')
    
    # Training/Optimization parameters
    parser.add_argument('--batch_size', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument('--pretrain_epochs', default=4, type=int, help='Number of epochs of finetuning the ViT.')
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate""")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="""Initial value of the
        weight decay.""")
    
    # Misc
    parser.add_argument('--dataset_name', default='CBMNIST', type=str,
        choices=['CBMNIST', 'COCOCOLOURS', 'WATERBIRDS'],
        help="""Dataset""")
    parser.add_argument('--data_path', type=str,
        help="""Dataset path""")
    parser.add_argument('--log', default=True, type=bool, help="""Output in log.txt or cmd""")
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument("--visualization_path", type=str, default="./vis", help="""Path to save visualizations""")
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')

    return parser


def get_dataloaders(args):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), antialias=False),
        # transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), antialias=False),
        # transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainloader = get_loader(args, 'train', transform=transform_train)
    testloader = get_loader(args, 'test', transform=transform_test)
    return trainloader, testloader

def pretrain(args, trainloader, testloader):
    # optimizer
    vit_optimizer = torch.optim.Adam(inv_rat.predictor_params(), lr=args.lr)

    global_step = 0
    for epoch in range(args.pretrain_epochs):
        print("=========================")
        print("epoch:", epoch)
        print("=========================")
        global_step = train_vit(trainloader, inv_rat, vit_optimizer, None, global_step, args)
        # dev
        with torch.no_grad():
            inv_acc = test_vit(args, testloader, inv_rat)

def train(args, trainloader, testloader):
    # Freeze Vit weights
    inv_rat.freeze_vit()
    inv_rat.freeze_classifier_head()

    optimizer = torch.optim.Adam(inv_rat.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, mode = 'exp_range', gamma=0.8, cycle_momentum=False)

    global_step = 0

    # learning
    dev_results = []
    te_results = []
    for epoch in range(args.epochs):
        print("=========================")
        print("epoch:", epoch)
        print("=========================")
        global_step = train_mnist(trainloader, inv_rat, optimizer, scheduler, global_step, args)
        # dev
        with torch.no_grad():
            inv_acc = test_mnist(args, testloader, inv_rat)
            dev_results.append(inv_acc.cpu().detach())
        # ============ writing logs ... ============
        save_dict = {
            'model': inv_rat.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args
        }
        torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))    

    ############################
    #visualization
    ############################
    if not os.path.exists(args.visualization_path):
        os.makedirs(args.visualization_path)
    transfrom = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), antialias=False),
        # transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transfrom_resize = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), antialias=False),
    ])
    dataset = get_dataset (args, phase = 'test', transform=transfrom)
    images = dataset.images
    labels = dataset.labels

    
    sample_labels = []
    for i in range (args.num_classes):
      sample_labels.append(images[labels[:,i]==1])
    
    
    sample_images = np.random.choice(len(images)//20, 10)
    inv_rat.eval()
    
    image_num = 0
    model = inv_rat
    for j in range (args.num_classes):
      for i,index in enumerate(sample_images):
          
        image = torch.unsqueeze(sample_labels[j][index],0).to(device)
        
        rationales, logits = inv_rat(transfrom(image))
        visualize (args.visualization_path, model, transfrom_resize(image), rationales, logits, j, patch_size=args.patch_size, image_size = args.img_size, show_cam=False, id=f'image{image_num}')
        image_num += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VitInvRat', parents=[get_args_parser()])
    args = parser.parse_args()
    
    set_seed(args.seed)
    if args.log:
        sys.stdout = open("log.txt","w")

    
    
    # build the model
    inv_rat = InvRat(args).to(device)
    for n,p in inv_rat.named_parameters():
        print(n)
    
    # Data
    args.batch_size = 128
    trainloader, testloader = get_dataloaders(args)
    # pretrain
    pretrain(args, trainloader, testloader)
    # Data
    args.batch_size = 32
    trainloader, testloader = get_dataloaders(args)
    # train
    train(args, trainloader, testloader)
    