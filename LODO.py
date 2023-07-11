#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import hickle as hkl 
import numpy as np
import os
import warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# In[ ]:


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
randomSeed = 0
np.random.seed(randomSeed)


# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


mainDir = './Datasets'


# In[ ]:


datasetList = ['HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'] 


# In[ ]:


dirName = 'SSL_PipelineUnion/LODO'
os.makedirs(dirName, exist_ok=True)


# In[ ]:


fineTuneDir = 'fineTuneData'
testDir = 'testData'
valDir = 'valData'
datasetDir = 'datasets'
os.makedirs(dirName+'/'+datasetDir, exist_ok=True)
os.makedirs(dirName+'/'+fineTuneDir, exist_ok=True)
os.makedirs(dirName+'/'+testDir, exist_ok=True)
os.makedirs(dirName+'/'+valDir, exist_ok=True)


# In[ ]:


for datasetIndex,dataSetName in enumerate(datasetList):
    datasetLabel = hkl.load(mainDir + 'datasetClientsUnion/'+dataSetName+'/clientsLabel.hkl')
    datasetTrain = hkl.load(mainDir + 'datasetClientsUnion/'+dataSetName+'/clientsData.hkl')
    print("Dataset: "+str(dataSetName) +" Shape: " + str(np.unique(np.argmax(np.vstack((datasetLabel)),axis = -1 ))))


# In[ ]:


fineTuneData = []
fineTuneLabel = []

for datasetIndex,dataSetName in enumerate(datasetList):
    datasetLabel = hkl.load(mainDir + 'datasetClientsUnion/'+dataSetName+'/clientsLabel.hkl')
    datasetTrain = hkl.load(mainDir + 'datasetClientsUnion/'+dataSetName+'/clientsData.hkl')
    hkl.dump(datasetTrain,dirName+'/'+datasetDir+ '/'+dataSetName+'_data.hkl')
    hkl.dump(datasetLabel,dirName+'/'+datasetDir+ '/'+dataSetName+'_label.hkl')
    trainingData = []
    testingData = []
    validatingData = []
    
    trainingLabel = []
    testingLabel = []
    validatingLabel = []
    
    for datasetData, datasetLabels in zip(datasetTrain,datasetLabel):
        nonSoftMaxedLabels = np.argmax(datasetLabels,axis = -1)
        
        skf = StratifiedKFold(n_splits=10,shuffle = False)
        skf.get_n_splits(datasetData, nonSoftMaxedLabels)
        partitionedData = list()
        partitionedLabel = list()
        testIndex = []
        
        for train_index, test_index in skf.split(datasetData, nonSoftMaxedLabels):
            testIndex.append(test_index)

        trainIndex = np.hstack((testIndex[:7]))
        devIndex = testIndex[8]
        testIndex = np.hstack((testIndex[8:]))

        X_train = tf.gather(datasetData,trainIndex).numpy()
        X_val = tf.gather(datasetData,devIndex).numpy()
        X_test = tf.gather(datasetData,testIndex).numpy()

        y_train = tf.gather(nonSoftMaxedLabels,trainIndex).numpy()
        y_val = tf.gather(nonSoftMaxedLabels,devIndex).numpy()
        y_test = tf.gather(nonSoftMaxedLabels,testIndex).numpy()
        
        y_train = tf.one_hot(y_train,10)
        y_val = tf.one_hot(y_val,10)
        y_test = tf.one_hot(y_test,10)

        trainingData.append(X_train)
        validatingData.append(X_val)
        testingData.append(X_test)
        
        trainingLabel.append(y_train)
        validatingLabel.append(y_val)
        testingLabel.append(y_test)
        
        
    testingLabel = np.asarray(testingLabel)
    testingData = np.asarray(testingData)
    
    validatingData = np.asarray(validatingData)
    validatingLabel = np.asarray(validatingLabel)

    trainingLabel = np.asarray(trainingLabel)
    trainingData = np.asarray(trainingData)
    
    fineTuneData.append(trainingData)
    fineTuneLabel.append(trainingLabel)

    hkl.dump(trainingData,dirName+'/'+fineTuneDir+ '/'+dataSetName+'_70_data.hkl')
    hkl.dump(trainingLabel,dirName+'/'+fineTuneDir+ '/'+dataSetName+'_70_label.hkl')
    hkl.dump(testingData,dirName+'/'+testDir+ '/'+dataSetName+'_data.hkl' )
    hkl.dump(testingLabel,dirName+'/'+testDir+ '/'+dataSetName+'_label.hkl' )
    hkl.dump(validatingData,dirName+'/'+valDir+ '/'+dataSetName+'_data.hkl' )
    hkl.dump(validatingLabel,dirName+'/'+valDir+ '/'+dataSetName+'_label.hkl' )
fineTuneData = np.asarray(fineTuneData)
fineTuneLabel = np.asarray(fineTuneLabel)


# In[ ]:


fineTuneRatio = [1.0,5.0,10.0]


# In[ ]:


fineTuneData = np.asarray(fineTuneData)
fineTuneLabel = np.asarray(fineTuneLabel)


# In[ ]:


meanSamplesUserDataset = []
for ratio in fineTuneRatio:
    datasetIndex = 0
    for trainingDataSubject, traningLabelSubject in zip(fineTuneData,fineTuneLabel):
        trainingDataSave = []
        trainingLabelSave = []
        sampleCount = []
        leaveOutRatio = (70.0 - ratio)/70.0
        print("Processing "+str(datasetList[datasetIndex]) + " with ratio: "+str(ratio))
        for subjectData, subjectLabel in zip(trainingDataSubject,traningLabelSubject):
            softMaxedLabels = np.argmax(subjectLabel,axis = -1)        
            if(((1-leaveOutRatio) * len(softMaxedLabels)) < 10):
                print("1% ratio unable to get 1 sample of each activity in "+str(datasetList[datasetIndex]) +" dataset")
                classIndex = np.unique(softMaxedLabels,return_index = True)[1]
                y_train = tf.one_hot(tf.gather(softMaxedLabels,classIndex).numpy(),10)
                trainingDataSave.append(tf.gather(subjectData,classIndex).numpy())
                trainingLabelSave.append(y_train)    
                sampleCount.append(trainingLabelSave[-1].shape[0])
            else:
                smallClass = np.where(np.unique(softMaxedLabels,return_counts=True)[1] < 10 )
                print(smallClass)
                if(len(smallClass) > 0):
                    for classToRemove in smallClass:
                        print("To remove index due to too few classes")
                        indicesRemove = np.where(softMaxedLabels == classToRemove)[0]
                        print(indicesRemove)
                        softMaxedLabels = np.delete(softMaxedLabels, indicesRemove,axis = 0)
                        subjectData = np.delete(subjectData, indicesRemove,axis = 0)
                X_train, X_val_test, y_train, y_val_test = train_test_split(subjectData, softMaxedLabels,
                                                            stratify=softMaxedLabels, 
                                                            random_state = randomSeed,
                                                            test_size=leaveOutRatio)
                y_train = tf.one_hot(y_train,10)
                trainingDataSave.append(X_train)
                trainingLabelSave.append(y_train)    
                sampleCount.append(trainingLabelSave[-1].shape[0])

        trainingLabelSave = np.asarray(trainingLabelSave)
        trainingDataSave = np.asarray(trainingDataSave)
        
        print("Dataset : "+str(datasetList[datasetIndex]))
        print(np.unique(np.argmax(np.vstack((trainingLabelSave)),-1), return_counts = True))
        
        meanSampleCount = int(np.mean(sampleCount))
        print("Mean samples per dataset is "+str(meanSampleCount))
        meanSamplesUserDataset.append(meanSampleCount)
        hkl.dump(trainingDataSave,dirName+'/'+fineTuneDir+ '/'+datasetList[datasetIndex]+'_'+str(int(ratio))+'_data.hkl')
        hkl.dump(trainingLabelSave,dirName+'/'+fineTuneDir+ '/'+datasetList[datasetIndex]+'_'+str(int(ratio))+'_label.hkl')
        datasetIndex += 1


# In[ ]:




