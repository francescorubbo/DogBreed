#import pandas as pd
#sample_submission = pd.read_csv('data/sample_submission.csv')
#print( sample_submission )

NUM_CLASSES = 120
MODELNAME = 'resnet152'

import torch
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()
model = torch.load('models/'+MODELNAME+'_nclasses%s.pth'%NUM_CLASSES)

import glob
ids = []
for imgname in glob.glob('data/test/dogs/*'):
    id = imgname.split('/')[-1].split('.jpg')[0]
    ids.append( id )
total = len(ids)

import torchvision
from dataprovider import gettrans
transform = gettrans()
datadir = '/home/rubbo/CNNexercise/dogbreed/data/test/'
image_data = torchvision.datasets.ImageFolder(datadir,transform=transform)
data_loader = torch.utils.data.DataLoader(image_data,
                                          batch_size=512,
                                          shuffle=False)

print( 'making predictions' )
toprob = torch.nn.Softmax()
preds = []
for jj,(inputs,_) in enumerate(data_loader):
    if use_gpu:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)

    outputs = model(inputs)
    outputs = toprob(outputs)
    print( jj*512,'/',total )
    preds.append( outputs )

print( 'done predicting' )
preds = torch.cat(preds)
preds = preds.data.cpu().numpy()
print( preds.shape )

import pandas as pd
df = pd.read_pickle('testlabels_120.pkl')
breed = df.keys().levels[1].values[:-1]
df = pd.DataFrame(index=ids, columns=breed, data=preds)
df.index.name = 'id'
df.to_csv('testsubmission.csv')


