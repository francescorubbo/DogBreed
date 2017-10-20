NUM_CLASSES = 120
MODELNAME = 'resnet152'

import torchvision.models as models
modelsdict = {}
modelsdict['resnet18'] = models.resnet18(pretrained=True)
modelsdict['resnet152'] = models.resnet152(pretrained=True)
#squeezenet = models.squeezenet1_1(pretrained=True)
#inception = models.inception_v3(pretrained=True)

model = modelsdict[MODELNAME]

from dataprovider import getdata
train_dl,valid_dl = getdata(NUM_CLASSES)

from training import getmodel
model = getmodel(NUM_CLASSES,model,train_dl,valid_dl)

import torch
torch.save(model,'models/'+MODELNAME+'_nclasses%s.pth'%NUM_CLASSES)
