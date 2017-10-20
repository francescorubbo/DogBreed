import pandas as pd

from utils import DogsDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def loadlabels(num_classes=120):
    labels = pd.read_csv('data/labels.csv')
    
    selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index)
    labels = labels[labels['breed'].isin(selected_breed_list)]
    labels['rank'] = labels.groupby('breed').rank()['id']
    labels_pivot = labels.pivot('id', 'breed').reset_index().fillna(0)
    return labels_pivot

def gettrans():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ds_trans = transforms.Compose([transforms.Scale(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize])
    return ds_trans
    
def getdata(num_classes=120):
    labels_pivot = loadlabels(num_classes)
    test = pd.read_pickle('testlabels_%s.pkl'%num_classes)
    labels_pivot = labels_pivot[~labels_pivot['id'].isin(test['id'])]
    
    train = labels_pivot.sample(frac=0.8)
    valid = labels_pivot[~labels_pivot['id'].isin(train['id'])]
    print(train.shape, valid.shape, test.shape)
    
    ds_trans = gettrans()

    train_ds = DogsDataset(train, 'data/train/', transform=ds_trans)
    valid_ds = DogsDataset(valid, 'data/train/', transform=ds_trans)
    
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=True, num_workers=4)

    return train_dl,valid_dl


def gettestdata(num_classes=120):
    test = pd.read_pickle('testlabels_%s.pkl'%num_classes)
    ds_trans = gettrans()
    test_ds = DogsDataset(test, 'data/train/', transform=ds_trans)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    return test_ds,test_dl

def definetest(num_classes=120,testsize=0.2):
    labels_pivot = loadlabels(num_classes)
    test = labels_pivot.sample(frac=testsize)
    test.to_pickle('testlabels_%s.pkl'%num_classes)
