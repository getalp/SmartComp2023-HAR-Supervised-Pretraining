#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import hickle as hkl 
import tensorflow as tf
import seaborn as sns
import logging
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def projectTSNEWithShape(fileName,filepath,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels,globalPrototypesIndex = None ):
    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=labels_argmax,
        style = labels_argmax,
        palette=sns.color_palette(n_colors = len(unique_labels)),
        s=90,
        alpha=1.0,
        rasterized=True,
        markers = True
    )
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(ACTIVITY_LABEL[int(label)]) 
        

    plt.tick_params(
    axis='both',         
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,        
    labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if(globalPrototypesIndex != None):
        plt.scatter(tsne_projections[globalPrototypesIndex,0],tsne_projections[globalPrototypesIndex,1], s=400,linewidth=3, facecolors='none', edgecolor='black')
    plt.savefig(filepath+fileName+".svg", bbox_inches="tight", format="svg")
    plt.show()
    plt.clf()


def projectTSNE(fileName,filepath,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels,globalPrototypesIndex = None ):
    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=labels_argmax,
        palette=sns.color_palette(n_colors = len(unique_labels)),
        s=90,
        alpha=1.0,
        rasterized=True
    )
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(ACTIVITY_LABEL[int(label)]) 
        

    plt.tick_params(
    axis='both',         
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,        
    labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if(globalPrototypesIndex != None):
        plt.scatter(tsne_projections[globalPrototypesIndex,0],tsne_projections[globalPrototypesIndex,1], s=400,linewidth=3, facecolors='none', edgecolor='black')
    plt.savefig(filepath+fileName+".svg", bbox_inches="tight", format="svg")
    plt.show()
    plt.clf()
def projectTSNEWithPosition(dataSetName,fileName,filepath,ACTIVITY_LABEL,labels_argmax,orientationsNames,clientOrientationTest,tsne_projections,unique_labels):
    classData = [ACTIVITY_LABEL[i] for i in labels_argmax]
    orientationData = [orientationsNames[i] for i in np.hstack((clientOrientationTest))]
    if(dataSetName == 'REALWORLD_CLIENT'):
        orientationName = 'Position'
    else:
        orientationName = 'Device'
    pandaData = {'col1': tsne_projections[:,0], 'col2': tsne_projections[:,1],'Classes':classData, orientationName :orientationData}
    pandaDataFrame = pd.DataFrame(data=pandaData)

    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    sns.scatterplot(data=pandaDataFrame, x="col1", y="col2", hue="Classes", style=orientationName,
                    palette=sns.color_palette(n_colors = len(unique_labels)),
                    s=90, alpha=1.0,rasterized=True,)
    plt.tick_params(
    axis='both',          
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,       
    labelbottom=False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath+fileName+".svg", bbox_inches="tight", format="svg")
    plt.show()
    plt.clf()

def plot_learningCurve(history, epochs, filepath, title = ""):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'],markevery=[np.argmax(history.history['val_accuracy'])], ls="", marker="o",color="orange")
    plt.plot(epoch_range, history.history['accuracy'],markevery=[np.argmax(history.history['accuracy'])], ls="", marker="o",color="blue")

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.savefig(filepath+title+"LearningAccuracy.png", bbox_inches="tight")
    plt.show()
    plt.clf()
    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.plot(epoch_range, history.history['loss'],markevery=[np.argmin(history.history['loss'])], ls="", marker="o",color="blue")
    plt.plot(epoch_range, history.history['val_loss'],markevery=[np.argmin(history.history['val_loss'])], ls="", marker="o",color="orange")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(filepath+title+"ModelLoss.png", bbox_inches="tight")
    plt.show()
    plt.clf()


def roundNumber(toRoundNb):
    return round(toRoundNb, 4) * 100
def converTensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer-1].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model


def loadFineTuneData(trainingRatio,testingDataset,evaluationType,dataDir):
    fineTuneData = hkl.load(dataDir + 'fineTuneData/'+testingDataset+'_'+str(trainingRatio)+'_data.hkl')
    fineTuneLabel = hkl.load(dataDir + 'fineTuneData/'+testingDataset+'_'+str(trainingRatio)+'_label.hkl')
    if(evaluationType == 'group'):
        fineTuneData = np.vstack((fineTuneData))
        fineTuneLabel = np.vstack((fineTuneLabel))
    return fineTuneData,fineTuneLabel


# def loadFineTuneData(trainingRatio,testingDataset,experimentSetting,evaluationType,dataDir,datasetList):
#     if(experimentSetting == 'LODO'):
#         fineTuneData = hkl.load(dataDir + 'fineTuneData/'+testingDataset+'_'+str(trainingRatio)+'_data.hkl')
#         fineTuneLabel = hkl.load(dataDir + 'fineTuneData/'+testingDataset+'_'+str(trainingRatio)+'_label.hkl')
#     else:
#         fineTuneData = []
#         fineTuneLabel = []
#         for datasetName in datasetList:
#             fineTuneData.append(hkl.load(dataDir + 'fineTuneData/'+datasetName+'_'+str(trainingRatio)+'_data.hkl'))
#             fineTuneLabel.append(hkl.load(dataDir + 'fineTuneData/'+datasetName+'_'+str(trainingRatio)+'_label.hkl'))
#         fineTuneData = np.asarray(fineTuneData)
#         fineTuneLabel = np.asarray(fineTuneLabel)
#     if(evaluationType == 'group'):
#         if(experimentSetting == 'LODO'):
#             fineTuneData = np.vstack((fineTuneData))
#             fineTuneLabel = np.vstack((fineTuneLabel))
#         else:
#             fineTuneData = [np.vstack((data)) for data in fineTuneData]
#             fineTuneLabel = [np.vstack((label)) for label in fineTuneLabel]
#             fineTuneData = np.asarray(fineTuneData)
#             fineTuneLabel = np.asarray(fineTuneLabel)
#     return fineTuneData,fineTuneLabel
