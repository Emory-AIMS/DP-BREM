import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# to split the dataset into non-iid subset
from torch.utils.data import Subset 
# to assemble different existing datasets (a list of Dataset) 

from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.grad_sample import GradSampleModule

import yaml


class Attacker:
    train_loader = None # training data with benign records
    train_loader_poison = None # training data with poisoned records
    test_loader = None # with poisoned labeles (only for )
    model = None
    optimizer = None
    benign_momentums = None


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.tanh(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.tanh(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.tanh(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def load_data_mnist(args):
    """
    MNIST dataset: This function loads a list of train_loaders (each for one client) and one test_loader
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # idx_attacker is the index of malicious clients's data
    idx_attacker= np.array([]).astype(int)
    attacker= Attacker()

    data_root = "../../data/mnist"
    test_batch_size = 1000
    num_clients = args.num_clients


    shard_per_client = 5
    num_shards = shard_per_client*num_clients
    shards = np.arange(num_shards)
    shard_size = int(60000 / num_shards)
    np.random.shuffle(shards)

    train_data = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    idx_sort = np.argsort(train_data.targets.numpy())
    train_data_split = []
    for i in range(num_clients):
        my_shards = shards[(i * shard_per_client) : (i + 1) * shard_per_client]
        idx = np.array([]).astype(int)
        labels = np.array([]).astype(int) # only for print purpose
        for j in my_shards:
            select = idx_sort[(j * shard_size): (j + 1) * shard_size]
            idx = np.concatenate((idx, select), axis=None)
            labels = np.concatenate((labels, train_data.targets[idx].numpy()), axis=None)

        train_data_split.append(Subset(train_data, idx))
        # print("Client {} gets data with label {}".format(i, np.unique(labels)))

        # We let the first num_bad_clients clients as the malicious ones
        if i < args.num_bad_clients:
            idx_attacker= np.concatenate((idx_attacker, idx), axis=None)


    # Load training data (a list of DataLoader, where each for one client)
    train_loaders = [torch.utils.data.DataLoader(
        x, batch_sampler=UniformWithReplacementSampler(
            num_samples=len(x),
            sample_rate=args.record_sampled_rate)
        ) for x in train_data_split]
    
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_root, train=False, transform=transform),
        batch_size = test_batch_size,
        shuffle=True
    )


    
    # Load attacker training data and test data 
    if args.num_bad_clients > 0:
        attacker_batch_size = 256

         # training data: benign records
        attacker.train_loader = torch.utils.data.DataLoader(
            Subset(train_data, idx_attacker),
            batch_size = attacker_batch_size,
            shuffle=True 
        )

        if args.attack_type ==  "backdoor": 

            # training data: poisoned records (triggered images and modified labels)
            train_data_poison = datasets.MNIST(data_root, train=True, download=True, transform=transform)
            train_data_poison.data[:, -2:, -2:] = 255
            train_data_poison.targets[:] = 0
            attacker.train_loader_poison = torch.utils.data.DataLoader(
                Subset(train_data_poison, idx_attacker),
                batch_size = attacker_batch_size,
                shuffle=True 
            )

            # test data: poisoned records (triggered images and modified labels)
            backdoor_test_data = datasets.MNIST(data_root, train=False, transform=transform)
            index_list = np.where(np.array(backdoor_test_data.targets) != 0)[0]
            backdoor_test_data.data[:, -2:, -2:] = 255
            backdoor_test_data.targets[:] = 0
            attacker.test_loader = torch.utils.data.DataLoader(
                Subset(backdoor_test_data, index_list), # only for images with true label not 0
                batch_size = test_batch_size,
                shuffle=True
            )


    return train_loaders, test_loader, attacker


"""
def load_data_mnist_ardis_backdoor(args):
    
    # MNIST dataset: This function loads a list of train_loaders (each for one client) and one test_loader

    def get_ardis_dataset(data_root, transform, train=True):
        
        # The raw ARDIS data can be downloaded (and then unrar) from:
        # https://raw.githubusercontent.com/ardisdataset/ARDIS/master/ARDIS_DATASET_IV.rar
        
        # load the data from csv's
        filename_prefix = "../../data/ARDIS/ARDIS_"
        if train:
            filename_prefix += "train_"
        else:
            filename_prefix += "test_"
        ardis_images=np.loadtxt(filename_prefix+'2828.csv', dtype='float')
        ardis_labels=np.loadtxt(filename_prefix+'labels.csv', dtype='float')

        #### reshape to be [samples][width][height]
        ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')

        # labels are one-hot encoded
        indices_seven = np.where(ardis_labels[:,7] == 1)[0]
        images_seven = ardis_images[indices_seven,:]
        images_seven = torch.tensor(images_seven).type(torch.uint8)

        ardis_dataset = datasets.MNIST(data_root, train=train, download=True, transform=transform)
        ardis_dataset.data = images_seven
        ardis_dataset.targets = torch.ones(len(indices_seven)).type(torch.int)  # modify lable 7 to label 1

        return ardis_dataset

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


    # idx_attacker is the index of malicious clients's data, which includes semantic poisoned images and other benign images
    idx_attacker = np.array([]).astype(int)

    data_root = "../../data/mnist"
    num_clients = args.num_clients
    train_data = datasets.MNIST(data_root, train=True, download=True, transform=transform)

    # random shard for each client (non-iid for federated learning)
    shard_per_client = 6
    num_shards = shard_per_client*num_clients
    shards = np.arange(num_shards)
    shard_size = int(60000 / num_shards)
    np.random.shuffle(shards)
    
    idx_sort = np.argsort(train_data.targets.numpy())
    train_data_split = []
    for i in range(num_clients):
        my_shards = shards[(i * shard_per_client) : (i + 1) * shard_per_client]
        idx = np.array([]).astype(int)
        for j in my_shards:
            select = idx_sort[(j * shard_size): (j + 1) * shard_size]
            idx = np.concatenate((idx, select), axis=None)

        train_data_subset = datasets.MNIST(data_root, train=True, download=False, transform=transform)
        train_data_subset.data = train_data_subset.data[idx]
        train_data_subset.targets = train_data_subset.targets[idx]
        train_data_split.append(train_data_subset)


        # We let the first num_bad_clients clients as the malicious ones
        if i < args.num_bad_clients:
            idx_attacker= np.concatenate((idx_attacker, idx), axis=None)

    # Load training data (a list of DataLoader, where each for one client)
    train_loaders = [torch.utils.data.DataLoader(
        x, batch_sampler=UniformWithReplacementSampler(
            num_samples=len(x),
            sample_rate=args.record_sampled_rate)
        ) for x in train_data_split]
    
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_root, train=False, transform=transform),
        batch_size = 1024,
        shuffle=True
    )


    
    attacker= Attacker()
    # Load attacker training data and test data 
    if args.num_bad_clients > 0:
        attacker_batch_size = 256

         # training data: benign records
        attacker.train_loader = torch.utils.data.DataLoader(
            Subset(train_data, idx_attacker),
            batch_size = attacker_batch_size,
            shuffle=True 
        )

        if args.attack_type ==  "backdoor": 

            # training data: poisoned records (triggered images and modified labels)
            train_data_poison = get_ardis_dataset(data_root, transform, train=True)
            attacker.train_loader_poison = torch.utils.data.DataLoader(
                train_data_poison,
                batch_size = attacker_batch_size,
                shuffle=True 
            )

            # test data: poisoned records (triggered images and modified labels)
            # even when no backdoor attack, we still can test the backdoor task (just for comparison purpose)
            backdoor_test_data = get_ardis_dataset(data_root, transform, train=True)
            attacker.test_loader = torch.utils.data.DataLoader(
                backdoor_test_data, 
                batch_size = len(backdoor_test_data.data),
                shuffle=True
            )

    return train_loaders, test_loader, attacker
"""



def load_data_cifar(args):
    """
    CIFAR10 dataset: This function loads a list of train_loaders (each for one client) and one test_loader
    """

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    with open("../../data/config/cifar_backdoor.yaml", "r") as f: 
        config = yaml.safe_load(f)

    # idx_attackeris the index of malicious clients's data
    # which includes semantic poisoned images and other benign images
    idx_attacker= np.array([]).astype(int) 
    idx_poison_train = config["background_wall"]        # index of poisoned images in the training data
    idx_poison_test = config["background_wall_test"]    # index of poisoned images in the test data
    attacker= Attacker()

    data_root = "../../data/cifar10"
    test_batch_size = 1000
    num_clients = args.num_clients
    data_per_client = int(50000 / num_clients) 

    train_data = datasets.CIFAR10(data_root, train=True, download=True, transform=transform_train)
    targets = np.array(train_data.targets)
    train_data_split = []

    if args.attack_type ==  "backdoor": 
        # label poison images as -1, which makes benign clients do not have these
        targets[idx_poison_train] = -1 
        


    for i in range(num_clients):
        # Dirichlet distribution with hyperparameter=0.9
        p_class = np.random.dirichlet(np.ones(10) * 0.9) 
        idx = np.array([]).astype(int)
        count = np.random.multinomial(data_per_client, p_class)
        # print("Client {} get data with label counts {}".format(i, count))

        for label in range(10):
            candidate = np.where(targets == label)[0]
            select = np.random.permutation(candidate)[: count[label]]
            idx = np.concatenate((idx, select), axis=None)
        train_data_split.append(Subset(train_data, idx))

        # We let the clients with index < num_bad_clients as the malicious ones
        if i < args.num_bad_clients:
            idx_attacker= np.concatenate((idx_attacker, idx), axis=None)

    # Load training data (a list of DataLoader, where each for one client)
    train_loaders = [torch.utils.data.DataLoader(
        x, batch_sampler=UniformWithReplacementSampler(
            num_samples=len(x),
            sample_rate=args.record_sampled_rate)
        ) for x in train_data_split]
    
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_root, train=False, transform=transform_test),
        batch_size = test_batch_size,
        shuffle=True
    )


    # Load attacker training data and test data 
    if args.num_bad_clients > 0:
        # training data: benign records
        attacker.train_loader = torch.utils.data.DataLoader(
            Subset(train_data, idx_attacker),
            batch_size = min(len(idx_poison_train)*10, len(idx_attacker)),
            shuffle=True
        )

        if args.attack_type ==  "backdoor":

            # training data: poisoned records (triggered images and modified labels)
            for j in idx_poison_train: 
                train_data.targets[j] = 2 # modify the label from car (class 1) to bird (class 2)
            attacker.train_loader_poison = torch.utils.data.DataLoader(
                Subset(train_data, idx_poison_train),
                batch_size = len(idx_poison_train),
                shuffle=True
            )

            # test data: poisoned records (attackerimages and modified labels)
            # we use randomly rotated and cropped versions (via "transform_train") 
            backdoor_test_data = datasets.CIFAR10(data_root, train=False, transform=transform_train)
            for j in idx_poison_test: 
                backdoor_test_data.targets[j] = 2
            
            attacker.test_loader = torch.utils.data.DataLoader(
                Subset(backdoor_test_data, idx_poison_test),
                batch_size = len(idx_poison_test)
            )


    return train_loaders, test_loader, attacker


def load_data(args):

    if args.name_dataset == "mnist":
        return load_data_mnist(args)
    elif args.name_dataset == "cifar":
        return load_data_cifar(args)
    else:
        print("The dataset should be 'mnist' or 'cifar'! ")
        raise NotImplementedError


def prepare_model(args, device):
    """
    This function returns a model with "GradSampleModule", which support per-sample gradient computation (see more from opacus)
    - We use the class 'GradSampleModule' from opacus package to efficiently implement per-example clipping (to guarantee record-level DP)
    - Therefore, we prepare models and datasets separately 
    """

    if args.name_dataset == "mnist":
        model = MnistCNN().to(device)
    elif args.name_dataset == "cifar":
        model = CifarCNN().to(device)
    else:
        print("The dataset should be either 'mnist' or 'cifar'! ")
        raise NotImplementedError

    return GradSampleModule(model, batch_first=True, loss_reduction="mean")


def prepare_data_model(args, device):

    train_loaders, test_loader, attacker = load_data(args)
    global_model = prepare_model(args, device)
    
    
    attacker.model = prepare_model(args, device)
    attacker.optimizer = optim.SGD(attacker.model.parameters(), lr=args.learn_rate, momentum=0)



    return train_loaders, test_loader, global_model, attacker



    