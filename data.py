import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torchvision import datasets, transforms
# to split the dataset into non-iid subset
from torch.utils.data import Subset, Dataset
# to assemble different existing datasets (a list of Dataset) 
import tensorflow as tf
import tensorflow_federated as tff
import random
import warnings
warnings.filterwarnings("ignore")

from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.grad_sample import GradSampleModule

import yaml


class Attacker:
    train_loader = None # training data with benign records
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

class FemnistCNN(nn.Module):
    def __init__(self):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
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
        batch_size = 1024,
        shuffle=True
    )


    
    # Load attacker training data and test data 
    if args.num_bad_clients > 0:

         # training data: benign records
        attacker.train_loader = torch.utils.data.DataLoader(
            Subset(train_data, idx_attacker),
            batch_size = 256,
            shuffle=True 
        )


    return train_loaders, test_loader, attacker





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


    data_root = "../../data/cifar10"
    num_clients = args.num_clients
    data_per_client = int(50000 / num_clients) 

    train_data = datasets.CIFAR10(data_root, train=True, download=True, transform=transform_train)
    targets = np.array(train_data.targets)
    train_data_split = []

    # idx_attackeris the index of malicious clients's data
    idx_attacker= np.array([]).astype(int) 


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
        batch_size = 1024,
        shuffle=True
    )


    # Load attacker training data and test data 
    attacker= Attacker()
    if args.num_bad_clients > 0:
        # training data: benign records
        attacker.train_loader = torch.utils.data.DataLoader(
            Subset(train_data, idx_attacker),
            batch_size = 256,
            shuffle=True
        )

    return train_loaders, test_loader, attacker


class TFDatasetToTorch(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = []
        for image, label in data:
            image = image.copy()
            label = label.copy().squeeze()
            label = torch.tensor(label, dtype=torch.long)
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    

def load_data_femnist(args):
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.02
    #session = tf.compat.v1.Session(config=config)
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_root = "../../data/femnist"
    num_clients = args.num_clients

    raw_train, raw_test = tff.simulation.datasets.emnist.load_data(cache_dir=data_root, only_digits=False)
    def train_preprocess(dataset):
        def batch_format_fn(element):
            return (tf.reshape(element['pixels'], [28, 28]), 
                    tf.reshape(element['label'], [1]))
        return dataset.map(batch_format_fn) 

    def test_preprocess(dataset):
        def batch_format_fn(element):
            return (tf.reshape(element['pixels'], [28, 28]), 
                    tf.reshape(element['label'], [1]))
        return dataset.map(batch_format_fn)
    
    raw_train = raw_train.preprocess(train_preprocess)
    raw_test = raw_test.preprocess(test_preprocess)

    trainSetList = []
    for cid in raw_train.client_ids[:num_clients]:
        data_train = list(raw_train.create_tf_dataset_for_client(cid).as_numpy_iterator())
        trainSetList.append(TFDatasetToTorch(data_train, transform=train_transform))
    random.shuffle(trainSetList)

    testSetList = []
    for cid in raw_test.client_ids[:num_clients]:
        data_test = list(raw_test.create_tf_dataset_for_client(cid).as_numpy_iterator())
        testSetList.append(TFDatasetToTorch(data_test, transform=test_transform))

    

    client_data_sizes = []

    train_loaders = []
    for subset in trainSetList:
        train_loader = torch.utils.data.DataLoader(
            subset, batch_sampler=UniformWithReplacementSampler(
            num_samples=len(subset),
            sample_rate=args.record_sampled_rate)
        )
        train_loaders.append(train_loader)
        client_data_sizes.append(len(subset))
    #print(f"client_data_sizes = {client_data_sizes}")

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(testSetList),
        batch_size = 1024,
        shuffle=True
    )

    attacker= Attacker()
    if args.num_bad_clients > 0:
        attacker.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(trainSetList[:args.num_bad_clients]),
            batch_size = 256,
            shuffle=True,
        )

    return train_loaders, test_loader, attacker


def load_data(args):

    if args.name_dataset == "mnist":
        return load_data_mnist(args)
    elif args.name_dataset == "cifar":
        return load_data_cifar(args)
    elif args.name_dataset == "femnist":
        return load_data_femnist(args)
    else:
        print("The dataset should be 'mnist' or 'cifar' or 'femnist'! ")
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
    elif args.name_dataset == "femnist":
        model = FemnistCNN().to(device)
    else:
        print("The dataset should be 'mnist' or 'cifar' or 'femnist'! ")
        raise NotImplementedError

    return GradSampleModule(model, batch_first=True, loss_reduction="mean")


def prepare_data_model(args, device):

    train_loaders, test_loader, attacker = load_data(args)
    global_model = prepare_model(args, device)



    return train_loaders, test_loader, global_model, attacker



    