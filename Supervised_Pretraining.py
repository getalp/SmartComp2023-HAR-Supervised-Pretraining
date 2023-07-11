#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
import hickle as hkl
import sklearn.manifold
import copy
import __main__ as main
import argparse
import pandas as pd
from tabulate import tabulate
from tensorflow import keras

seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


# Library scripts
import utils 
import training
import model


# In[ ]:


experimentSetting = 'LODO'

testingDataset = 'UCI'
# 'HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'
evaluationType = 'group'
# 'subject','group'

architecture = "ispl"
# "mobilehart_xs,hart,ispl,deepConvlstm"

# 1024,2048
finetune_epoch = 1

finetune_batch_size = 128

loss = 'Adam'
# 'LARS', 'Adam', 'SGD'
SSL_LR = 5e-4

input_shape = (128,6)

SSL_batch_size = 128

SSL_epochs = 200


# In[ ]:


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--architecture', type=str, default=architecture, 
                help='The choice of model architecture')  
    parser.add_argument('--testingDataset', type=str, default=testingDataset, 
            help='Left out dataset')  
    parser.add_argument('--evaluationType', type=str, default=evaluationType, 
        help='Dataset group evaluation or subject by subject evaluation')  
    parser.add_argument('--SSL_epochs', type=int, default=SSL_epochs, 
        help='SSL Epochs')  
    parser.add_argument('--SSL_batch_size', type=int, default=SSL_batch_size, 
        help='SSL batch_size')  
    parser.add_argument('--finetune_epoch', type=int, default=finetune_epoch, 
        help='Fine_tune Epochs')  
    parser.add_argument('--loss', type=str, default=loss, 
        help='Specify the loss') 
    parser.add_argument('--SSL_LR', type=float, default=SSL_LR, 
        help='Specify the learning rate for the SSL techniques') 

    args = parser.parse_args()
    return args
def is_interactive():
    return not hasattr(main, '__file__')


# In[ ]:


if not is_interactive():
    rootdir = './SSL_Project/'
    args = add_fit_args(argparse.ArgumentParser(description='Supervised Pre-training'))
    testingDataset = args.testingDataset
    evaluationType = args.evaluationType
    SSL_epochs = args.SSL_epochs
    SSL_batch_size = args.SSL_batch_size
    finetune_epoch = args.finetune_epoch
    loss = args.loss
    SSL_LR = args.SSL_LR
    architecture =  args.architecture
else:
    rootdir = '../../'

initWeightDir = rootdir+architecture+'_Weight.h5'


# In[ ]:


tf.keras.backend.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


# Dataset Metadata
# ACTIVITY_LABEL = []
ACTIVITY_LABEL = ['Downstairs', 'Upstairs','Running','Sitting','Standing','Walking','Lying','Cycling','Nordic_Walking','Jumping']
output_shape = len(ACTIVITY_LABEL)


# In[ ]:


dataDir = rootdir+'SSL_PipelineUnion/'+experimentSetting+'/'
projectName = 'supervised_'+str(finetune_epoch)+'/'+architecture+'_pretrain_epochs_'+str(SSL_epochs)
testMode = False
if(finetune_epoch < 10):
    testMode = True
    projectName= projectName + '/tests'
    
dataSetting = testingDataset

working_directory = rootdir+'results/'+projectName+'/'+experimentSetting+'/'+dataSetting+'/'
pretrained_dir = working_directory + evaluationType + '/'

os.makedirs(pretrained_dir, exist_ok=True)


# In[ ]:


datasetList = ["HHAR","MobiAct","MotionSense","RealWorld_Waist","UCI","PAMAP"] 


# In[ ]:


SSLdatasetList = copy.deepcopy(datasetList)
SSLdatasetList.remove(testingDataset)
SSL_data = []
SSL_label = []

SSL_val_data = []
SSL_val_label = []

for datasetName in SSLdatasetList:
    SSL_data.append(hkl.load(dataDir + 'testData/'+str(datasetName)+'_data.hkl'))
    SSL_data.append(hkl.load(dataDir + 'fineTuneData/'+str(datasetName)+'_70_data.hkl'))
    SSL_label.append(hkl.load(dataDir + 'testData/'+str(datasetName)+'_label.hkl'))
    SSL_label.append(hkl.load(dataDir + 'fineTuneData/'+str(datasetName)+'_70_label.hkl'))

    SSL_val_data.append(hkl.load(dataDir + 'valData/'+str(datasetName)+'_data.hkl'))
    SSL_val_label.append(hkl.load(dataDir + 'valData/'+str(datasetName)+'_label.hkl'))

SSL_data = np.vstack((np.hstack((SSL_data))))
SSL_label = np.vstack((np.hstack((SSL_label))))

SSL_val_data = np.vstack((np.hstack((SSL_val_data))))
SSL_val_label = np.vstack((np.hstack((SSL_val_label))))

testData = hkl.load(dataDir + 'testData/'+testingDataset+'_data.hkl')
testLabel = hkl.load(dataDir + 'testData/'+testingDataset+'_label.hkl')

valData = hkl.load(dataDir + 'valData/'+testingDataset+'_data.hkl')
valLabel = hkl.load(dataDir + 'valData/'+testingDataset+'_label.hkl')


# In[ ]:


testData = np.asarray([np.vstack((data)) for data in testData])
testLabel = np.asarray([np.vstack((label)) for label in testLabel])
valData = np.asarray([np.vstack((data)) for data in valData])
valLabel = np.asarray([np.vstack((label)) for label in valLabel])
testDataLength = [label.shape[0] for label in testLabel]


# In[ ]:


pretrain_base_save_path = working_directory+architecture+"_trained_supervised.h5"


# In[ ]:


if(architecture == "hart"):
    learningModel = model.HART(input_shape,output_shape)
elif(architecture == "ispl"):
    learningModel = model.ispl_inception(input_shape,output_shape)
elif(architecture == "deepConvlstm"):
    learningModel = model.deepConvLSTM((128,6,1),output_shape)
else:
    learningModel = model.mobilehart_xs(input_shape,output_shape)
    
if(not os.path.exists(initWeightDir)):
    print("model weights initialization not found, generating one")
    learningModel.save_weights(initWeightDir)
else:
    print("found initialization model weights")
    learningModel.load_weights(initWeightDir)
    
totalLayer = len(learningModel.layers)

if(architecture == "ispl_inception" or architecture == 'deepConvLSTM'):
    feature_extraction_layer = totalLayer - 2
else:
    feature_extraction_layer = totalLayer - 4


# In[ ]:


if(not os.path.exists(pretrain_base_save_path) or testMode):
    optimizer = tf.keras.optimizers.Adam(5e-4)
    learningModel.compile(
        optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"]
    )
    history = learningModel.fit(x = SSL_data,y = SSL_label, validation_data = (SSL_val_data,SSL_val_label),batch_size = SSL_batch_size, epochs = SSL_epochs,verbose=2)

    utils.plot_learningCurve(history,SSL_epochs,working_directory) 
    
    intermediate_model = model.extract_intermediate_model_from_base_model(learningModel,feature_extraction_layer)
    perplexity = 30.0
    embeddings = intermediate_model.predict(testData, batch_size=1024,verbose=0)
    tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=0, random_state=42)
    tsne_projections = tsne_model.fit_transform(embeddings)
    labels_argmax = np.argmax(testLabel, axis=1)
    unique_labels = np.unique(labels_argmax)
    utils.projectTSNE('TSNE_Embeds',pretrained_dir,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels )
    utils.projectTSNEWithShape('TSNE_Embeds_shape',pretrained_dir,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels )
    hkl.dump(tsne_projections,pretrained_dir+'tsne_projections.hkl')
    learningModel.save_weights(pretrain_base_save_path)
else:
    learningModel.load_weights(pretrain_base_save_path)
    optimizer = tf.keras.optimizers.Adam(5e-4)
    learningModel.compile(
        optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"]
    )
    print("Pre-trained model found, skipping training of pretrained model",flush = True)


# ### Downstream tasks

# In[ ]:


ratio = [1,5,10,70]


# In[ ]:


ratioResults = {}
for trainingRatio in ratio:
    print("Now downstreaming on "+testingDataset+" dataset with train ratio "+str(trainingRatio), flush = True)
    evaluation_dir = pretrained_dir+'ratio_'+ str(trainingRatio) + '/'
    os.makedirs(evaluation_dir, exist_ok=True)
    fineTuneData,fineTuneLabel = utils.loadFineTuneData(trainingRatio,testingDataset,evaluationType,dataDir)
    evaluationsF1 = training.downStreamPipeline(fineTuneData,fineTuneLabel,valData,valLabel,testData,testLabel,pretrain_base_save_path,evaluation_dir,initWeightDir,learningModel,
                                                output_shape = output_shape,
                                                finetune_epoch = finetune_epoch, 
                                                finetune_batch_size = finetune_batch_size, 
                                                feature_extraction_layer = feature_extraction_layer)
    ratioResults['ratio_'+str(trainingRatio)] = evaluationsF1


# In[ ]:


npRatio = np.asarray(list(ratioResults.values())).T
evaluationMethods = ['Transfer','Linear','Full','Full_Unfrozen','Full_Random']
ratioHeaders = ['ratio_'+str(rat) for rat in ratio]
ratioHeaders.insert(0, "dataset")
for evalIndex, methods in enumerate(evaluationMethods):
    toWriteEvaluation = {}
    toWriteEvaluation['dataset'] = [testingDataset]
    for ratioIndex, rat in enumerate(ratio):
        toWriteEvaluation['ratio_'+str(rat)] = [npRatio[evalIndex][ratioIndex]]
    tabular = tabulate(toWriteEvaluation, headers="keys")
    print(methods)
    print(tabular)
    print()
    text_file = open(pretrained_dir +methods+'_report.csv',"w")
    text_file.write(tabular)
    text_file.close()


# In[ ]:


allTrained = True 
for methods in evaluationMethods:
    print("Processing "+str(methods) +" report")
    fullReport = []
    ratioHeaders = ['ratio_'+str(rat) for rat in ratio]
    ratioHeaders.insert(0, "dataset")
    fullReport.append(ratioHeaders)
    if(allTrained):
        for datasetName in datasetList:
            checkDir = rootdir+'results/'+projectName+'/'+experimentSetting+'/'+datasetName+'/'+evaluationType+'/'+methods+"_report.csv"
            if(not os.path.exists(checkDir)):
                print("Dir below not found:")
                print(checkDir)
                allTrained = False
                break
            readData = pd.read_table(checkDir, delim_whitespace=True)
            fullReport.append(readData.to_numpy()[1])
    else:
        break
    if(allTrained):
        print("Generating "+str(methods) + " report")
        tabular2 = tabulate(fullReport)
        text_file = open(rootdir+'results/'+projectName+'/'+experimentSetting+'/'+str(evaluationType)+'_'+str(methods)+'_report.csv',"w")
        text_file.write(tabular2)
        text_file.close()
        print(tabular2)
        print()

