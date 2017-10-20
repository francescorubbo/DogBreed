NUM_CLASSES = 120
MODELNAME = 'resnet152'

import torch
model = torch.load('models/'+MODELNAME+'_nclasses%s.pth'%NUM_CLASSES)

from dataprovider import gettestdata
test_ds,test_dl = gettestdata(NUM_CLASSES)

import numpy as np
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()
preds = []
ys = []
for inputs,labels in test_dl:
#inputs,labels = next(iter(test_dl))
    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = model(inputs)
    toprob = torch.nn.Softmax()
    outputs = toprob(outputs)
#    preds.append( np.argmax(outputs.data.cpu().numpy()) )
    preds.append( outputs.data.cpu().numpy() )
    ys.append( labels.data.cpu().numpy()[0] )

import pandas as pd
testdf = pd.read_pickle('testlabels_%s.pkl'%NUM_CLASSES)
testdf['label'] = ys
testdf['output'] = preds
testdf.to_pickle('testdf_%s.pkl'%MODELNAME)

#df = pd.DataFrame({'label':ys,'output':preds})
#df.to_pickle('testdf.pkl')
