from torch import manual_seed
from torchvision import transforms
from torchvision import datasets
from torch.utils import data
from torchvision.transforms import ToTensor
import nonechucks as nc

def load_train_val_data(dir_,weighted_average = False):
    # print(dir_)
    train_dir = dir_ + '/train'
    val_dir = dir_ + '/val'

    train_transforms = transforms.Compose([transforms.Resize((128,128)),
                                           transforms.RandomRotation(90),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5832, 0.5695, 0.5371],[0.2093, 0.2163, 0.2292])
                                       ])

    val_transforms = transforms.Compose([transforms.Resize((128,128)),
                                          #transforms.RandomRotation(90),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5832, 0.5695, 0.5371],[0.2093, 0.2163, 0.2292])
                                      ])

    train_data = datasets.ImageFolder(train_dir,
                    transform=train_transforms)

    val_data = datasets.ImageFolder(val_dir,
                    transform=train_transforms)


    train_data = nc.SafeDataset(train_data)
    val_data = nc.SafeDataset(val_data)

    trainloader = data.DataLoader(train_data,
                   batch_size=32,shuffle=True,num_workers=0)

    if weighted_average:
        manual_seed(0)
        val_data_split = data.random_split(val_data,[len(val_data)//2,(len(val_data) - len(val_data)//2)])
        valloader_1 = data.DataLoader(val_data_split[0],
                       batch_size=32,shuffle=True,num_workers=0)
        valloader_2 = data.DataLoader(val_data_split[1],
                       batch_size=32,shuffle=True,num_workers=0)
        return trainloader, valloader_1, valloader_2

    else:
        valloader = data.DataLoader(val_data,
                       batch_size=32,shuffle=True,num_workers=0)

        return trainloader, valloader

def load_test_data(dir_):
    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

    test_transforms = transforms.Compose([transforms.Resize((128,128)),
                                          #transforms.RandomRotation(90),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5832, 0.5695, 0.5371],[0.2093, 0.2163, 0.2292])
                                      ])


    # EXAMPLE USAGE:
    # instantiate the dataset and dataloader
    data_dir = dir_ + '/test'
    dataset = ImageFolderWithPaths(data_dir,transform=test_transforms) # our custom dataset
    testloader = data.DataLoader(dataset)

    return testloader
