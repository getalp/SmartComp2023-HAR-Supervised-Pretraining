#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, LSTM, Reshape
from tensorflow.keras.models import Sequential

def deepConvLSTM(input_shape,activityCount):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 1), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=(5, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=(5, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=(5, 1)))
    model.add(Activation("relu"))
    model.add(Reshape((112, 6 * 64)))
    model.add(LSTM(128, activation="tanh", return_sequences=True))
    model.add(Dropout(0.5, seed=0))
    model.add(LSTM(128, activation="tanh"))
    model.add(Dropout(0.5, seed=1))
    model.add(Dense(activityCount, activation = 'softmax'))
    return model


class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x,training=None):
        if(training):
            input_shape = tf.shape(x)
            batch_size = input_shape[0]
            rank = x.shape.rank
            shape = (batch_size,) + (1,) * (rank - 1)
            random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
            path_mask = tf.floor(random_tensor)
            output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
            return output
        else:
            return x 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'drop_prob': self.drop_prob,})
        return config

class GatedLinearUnit(layers.Layer):
    def __init__(self,units,**kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
        self.linear = layers.Dense(units * 2)
        self.sigmoid = tf.keras.activations.sigmoid
    def call(self, inputs):
        linearProjection = self.linear(inputs)
        softMaxProjection = self.sigmoid(linearProjection[:,:,self.units:])
        return tf.multiply(linearProjection[:,:,:self.units],softMaxProjection)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,})
        return config
    

class ClassToken(layers.Layer):
    def __init__(self, hidden_size,**kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.hidden_size = hidden_size
        self.cls = tf.Variable(
            name="cls",
            initial_value=self.cls_init(shape=(1, 1, self.hidden_size), seed=randomSeed, dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,})
        return config

class Prompts(layers.Layer):
    def __init__(self, projectionDims,promptCount = 1,**kwargs):
        super(Prompts, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.projectionDims = projectionDims
        self.promptCount = promptCount
        self.prompts = [tf.Variable(
            name="prompt"+str(_),
            initial_value=self.cls_init(shape=(1, 1, self.projectionDims), seed=randomSeed, dtype="float32"),
            trainable=True,
        )  for _ in range(promptCount)]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        prompt_broadcasted = tf.concat([tf.cast(tf.broadcast_to(promptInits, [batch_size, 1, self.projectionDims]),dtype=inputs.dtype,)for promptInits in self.prompts],1)
        return tf.concat([inputs,prompt_broadcasted], 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionDims': self.projectionDims,
            'promptCount': self.promptCount,})
        return config
    
class SensorWiseMHA(layers.Layer):
    def __init__(self, projectionQuarter, num_heads,startIndex,stopIndex,dropout_rate = 0.0,dropPathRate = 0.0, **kwargs):
        super(SensorWiseMHA, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.MHA = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projectionQuarter, dropout = dropout_rate )
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.dropPathRate = dropPathRate
        self.DropPath = DropPath(dropPathRate)
    def call(self, inputData, training=None, return_attention_scores = False):
        extractedInput = inputData[:,:,self.startIndex:self.stopIndex]
        if(return_attention_scores):
            MHA_Outputs, attentionScores = self.MHA(extractedInput,extractedInput,return_attention_scores = True )
            return MHA_Outputs , attentionScores
        else:
            MHA_Outputs = self.MHA(extractedInput,extractedInput)
            MHA_Outputs = self.DropPath(MHA_Outputs)
            return MHA_Outputs
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'num_heads': self.num_heads,
            'startIndex': self.startIndex,
            'dropout_rate': self.dropout_rate,
            'stopIndex': self.stopIndex,
            'dropPathRate': self.dropPathRate,})
        return config
def softDepthConv(inputs):
    kernel = inputs[0]
    inputData = inputs[1]
    convOutputs = tf.nn.conv1d(
    inputData,
    kernel,
    stride = 1,
    padding = 'SAME',
    data_format='NCW',)
    return convOutputs


class liteFormer(layers.Layer):
    def __init__(self,startIndex,stopIndex, projectionSize, kernelSize = 16, attentionHead = 3, use_bias=False, dropPathRate = 0.0,dropout_rate = 0,**kwargs):
        super(liteFormer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.kernelSize = kernelSize
        self.softmax = tf.nn.softmax
        self.projectionSize = projectionSize
        self.attentionHead = attentionHead 
        self.dropPathRate = dropPathRate
        self.dropout_rate = dropout_rate
        self.DropPathLayer = DropPath(dropPathRate)
        self.projectionHalf = projectionSize // 2
    def build(self, input_shape):
        self.depthwise_kernel = [self.add_weight(
            shape=(self.kernelSize,1,1),
            initializer="glorot_uniform",
            trainable=True,
            name="convWeights"+str(_),
            dtype="float32") for _ in range(self.attentionHead)]
        if self.use_bias:
            self.convBias = self.add_weight(
                shape=(self.attentionHead,), 
                initializer="glorot_uniform", 
                trainable=True,  
                name="biasWeights",
                dtype="float32"
            )
        
    def call(self, inputs,training=None):
        formattedInputs = inputs[:,:,self.startIndex:self.stopIndex]
#         print(inputs.shape)
        inputShape = tf.shape(formattedInputs)
#         reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,self.projectionSize))
        reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,inputShape[1]))
        if(training):
            for convIndex in range(self.attentionHead):
                self.depthwise_kernel[convIndex].assign(self.softmax(self.depthwise_kernel[convIndex], axis=0))
        convOutputs = [tf.nn.conv1d(
            reshapedInputs[:,convIndex:convIndex+1,:],
            self.depthwise_kernel[convIndex],
            stride = 1,
            padding = 'SAME',
            data_format='NCW',) for convIndex in range(self.attentionHead) ]
        convOutputs = tf.convert_to_tensor(convOutputs)
        convOutputs = self.DropPathLayer(convOutputs)

        shape = tf.shape(formattedInputs)
        localAttention = tf.reshape(convOutputs,shape)
        return localAttention
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bias': self.use_bias,
            'patchCount': self.patchCount,
            'kernelSize': self.kernelSize,
            'startIndex': self.startIndex,
            'stopIndex': self.stopIndex,
            'projectionSize': self.projectionSize,
            'dropPathRate': self.dropPathRate,
            'dropout_rate': self.dropout_rate,
            'attentionHead': self.attentionHead,})
        return config          

class mixAccGyro(layers.Layer):
    def __init__(self,projectionQuarter,projectionHalf,projection_dim,**kwargs):
        super(mixAccGyro, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.projectionHalf = projectionHalf
        self.projection_dim = projection_dim
        self.projectionThreeFourth = self.projectionHalf+self.projectionQuarter
        self.mixedAccGyroIndex = tf.reshape(tf.transpose(tf.stack(
            [np.arange(projectionQuarter,projectionHalf), np.arange(projectionHalf,projectionHalf + projectionQuarter)])),[-1])
        self.newArrangement = tf.concat((np.arange(0,projectionQuarter),self.mixedAccGyroIndex,np.arange(self.projectionThreeFourth,projection_dim)),axis = 0)
    def call(self, inputs):
        return tf.gather(inputs,self.newArrangement,axis= 2)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'projectionHalf': self.projectionHalf,
            'projection_dim': self.projection_dim,
        })
        return config


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def mlp2(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0],activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    return x

def depthMLP(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0])(x)
    x = layers.DepthwiseConv1D(3,data_format='channels_first',activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

class SensorPatchesTimeDistributed(layers.Layer):
    def __init__(self, projection_dim,filterCount,patchCount,frameSize = 128, channelsCount = 6,**kwargs):
        super(SensorPatchesTimeDistributed, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.frameSize = frameSize
        self.channelsCount = channelsCount
        self.patchCount = patchCount
        self.filterCount = filterCount
        self.reshapeInputs = layers.Reshape((patchCount, frameSize // patchCount, channelsCount))
        self.kernelSize = (projection_dim//2 + filterCount) // filterCount
        self.accProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.gyroProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.flattenTime = layers.TimeDistributed(layers.Flatten())
        assert (projection_dim//2 + filterCount) / filterCount % self.kernelSize == 0
        print("Kernel Size is "+str((projection_dim//2 + filterCount) / filterCount))
#         assert 
    def call(self, inputData):
        inputData = self.reshapeInputs(inputData)
        accProjections = self.flattenTime(self.accProjection(inputData[:,:,:,:3]))
        gyroProjections = self.flattenTime(self.gyroProjection(inputData[:,:,:,3:]))
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projection_dim': self.projection_dim,
            'filterCount': self.filterCount,
            'patchCount': self.patchCount,
            'frameSize': self.frameSize,
            'channelsCount': self.channelsCount,})
        return config
    
class SensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(SensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:])
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config


class threeSensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(threeSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.magProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")

    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:6])
        magProjections = self.gyroProjection(inputData[:,:,6:])

        Projections = tf.concat((accProjections,gyroProjections,magProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

        
class fourSensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(fourSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.magProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.altProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")

    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:6])
        magProjection = self.gyroProjection(inputData[:,:,6:9])
        altProjection = self.gyroProjection(inputData[:,:,9:])

        Projections = tf.concat((accProjections,gyroProjections,magProjection,altProjection),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=4):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model

def HART_encoder(projection_dim = 192,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31],dropout_rate = 0.1,useTokens = False):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input((None, projection_dim))
    encoded_patches = inputs
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)
        
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return tf.keras.Model(inputs, representation, name="mae_encoder")


class PatchLayer(layers.Layer):
    def __init__(self, frameLength, frameStride,**kwargs):
        super(PatchLayer, self).__init__(**kwargs)
        self.frameLength = frameLength
        self.frameStride = frameStride

    def call(self, inputData, training=None):
        patchedData = tf.image.extract_patches(tf.expand_dims(inputData, 3), sizes=[1, self.frameLength, 1, 1], strides=[1, self.frameStride, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        patchedData = layers.Reshape((-1,patchedData.shape[2]*patchedData.shape[3]))(patchedData)
        return patchedData
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frameLength': self.frameLength,
            'frameStride': self.frameStride,})
        return config
    
    
class PatchDemo(layers.Layer):
    def __init__(self, frameLength, frameStride,**kwargs):
        super(PatchDemo, self).__init__(**kwargs)
        self.frameLength = frameLength
        self.frameStride = frameStride
    def call(self, inputData, training=None):
        patchedData = tf.image.extract_patches(tf.expand_dims(inputData, 3), sizes=[1, self.frameLength, 1, 1], strides=[1, self.frameStride, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        patchedData = tf.transpose(patchedData, [0, 1, 3, 2])
        return patchedData
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frameLength': self.frameLength,
            'frameStride': self.frameStride,})
        return config

def HART_decoder(enc_embedding_size,patch_count = 8, output_shape = (128,6), frame_length = 16, projection_dim = 192,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 7, 15],dropout_rate = 0.1,useTokens = False):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input((patch_count, enc_embedding_size))
    encoded_patches = layers.Dense(projection_dim)(inputs)
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)
        
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    pre_final = layers.Dense(units=output_shape[0] * output_shape[1])(representation)
    outputs = layers.Reshape(output_shape)(pre_final)
#     outputs = PatchLayer(frame_length,frame_length)(pre_final)
    return tf.keras.Model(inputs, outputs, name="mae_decoder")


def HART(input_shape,activityCount, projection_dim = 192,patchSize = 16,timeStep = 16,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024],dropout_rate = 0.3,useTokens = False):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input(shape=input_shape)
    patches = SensorPatches(projection_dim,patchSize,timeStep)(inputs)
    if(useTokens):
        patches = ClassToken(projection_dim)(patches)
    patchCount = patches.shape[1] 
    encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(
                          startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)

        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    if(useTokens):
        representation = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(representation)
    else:
        representation = layers.GlobalAveragePooling1D()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    logits = layers.Dense(activityCount,  activation='softmax')(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


# ------------------------------specific module for MobileHART------------------------------

def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv1D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)

def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv1D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding1D(padding=1)(m)
    m = layers.DepthwiseConv1D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    m = layers.Conv1D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    
    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m

def transformer_block(x, transformer_layers, projection_dim, dropout_rate = 0.3,droppath_rate = 0.3,num_heads=2):
    
    dropPathRate = np.linspace(0, droppath_rate* 10,transformer_layers) * 0.1
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp2(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=dropout_rate,
        )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x

def mobilevit_block(x, num_blocks, projection_dim, strides=1,dropout_rate = 0.3, droppath_rate = 0.3):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )
    global_features = transformer_block(
        local_features, num_blocks, projection_dim, dropout_rate = dropout_rate, droppath_rate = droppath_rate
    )
    
    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        global_features, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


def sensorWiseTransformer_block(xAcc, xGyro, patchCount,transformer_layers, projection_dim,kernelSize = 4, dropout_rate = 0.3, droppath_rate = 0.3,  num_heads=2):
    projectionQuarter = projection_dim // 4
    projectionHalf = projection_dim // 2
    dropPathRate = np.linspace(0, droppath_rate* 10,transformer_layers) * 0.1
    x = tf.concat((xAcc,xGyro),axis= 2 )
    for layerIndex in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6, name = "normalizedInputs_"+str(layerIndex))(x)

        branch1 = liteFormer( startIndex = projectionQuarter,
                              stopIndex = projectionQuarter + projectionHalf,
                              projectionSize = projectionHalf,
                              attentionHead =  num_heads, 
                              kernelSize = kernelSize,
                              dropPathRate = dropPathRate[layerIndex],
                              name = "liteFormer_"+str(layerIndex))(x1)

        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        # Skip connection 1.
        x2 = layers.Add()([concatAttention, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp2(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=dropout_rate,
        )
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x
def sensorWiseHART(xAcc,xGyro, num_blocks, projection_dim, kernelSize = 4, strides=1, dropout_rate = 0.3, droppath_rate = 0.3):
    # Local projection with convolutions.
#     ---------------acc--------------
    local_featuresAcc = conv_block(xAcc, filters=projection_dim//2, strides=strides)
    local_featuresAcc = conv_block(
        local_featuresAcc, filters=projection_dim//2, kernel_size=1, strides=strides
    )

#     ---------------gyro--------------

    local_featuresGyro = conv_block(xGyro, filters=projection_dim//2, strides=strides)
    local_featuresGyro = conv_block(
        local_featuresGyro, filters=projection_dim//2, kernel_size=1, strides=strides
    )
    global_features = sensorWiseTransformer_block(local_featuresAcc,
        local_featuresGyro, local_featuresGyro.shape[1], num_blocks, projection_dim, kernelSize = kernelSize, dropout_rate = dropout_rate,droppath_rate = droppath_rate
    )

    folded_feature_map_acc = conv_block(
        global_features[:,:,:projection_dim//2], filters=xAcc.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features_acc = layers.Concatenate(axis=-1)([xAcc, folded_feature_map_acc])
        # Fuse the local and global features using a convoluion layer.
    local_global_features_acc = conv_block(
        local_global_features_acc, filters=projection_dim//2, strides=strides
    )
    
    folded_feature_map_gyro = conv_block(
        global_features[:,:,projection_dim//2:], filters=xGyro.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features_gyro = layers.Concatenate(axis=-1)([xGyro, folded_feature_map_gyro])

    local_global_features_gyro = conv_block(
        local_global_features_gyro, filters=projection_dim//2, strides=strides
    )

    return local_global_features_acc, local_global_features_gyro
def mv2Block(x,expansion_factor,filterCount):
    x = inverted_residual_block(
        x, expanded_channels=filterCount[0] * expansion_factor, output_channels=filterCount[1]
    )
    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=filterCount[1] * expansion_factor, output_channels=filterCount[2], strides=2
    )
    x = inverted_residual_block(
        x, expanded_channels=filterCount[2] * expansion_factor, output_channels=filterCount[2]
    )
    x = inverted_residual_block(
        x, expanded_channels=filterCount[2] * expansion_factor, output_channels=filterCount[2]
    )
    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=filterCount[2] * expansion_factor, output_channels=filterCount[3], strides=2
    )
    return x
    # def hartModel(input_shape,activityCount, projection_dim,patchSize,timeStep,num_heads,filterAttentionHead, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024],dropout_rate = 0.3,useTokens = True):

def mobileHART_XS(input_shape,activityCount,projectionDims = [96,120,144],filterCount = [16//2,32//2,48//2,64//2,80,96,384],expansion_factor=4,mlp_head_units = [1024],dropout_rate = 0.3, droppath_rate = 0.3):
    
    # inputs = keras.Input((segment_size, num_input_channels))
    inputs = layers.Input(shape=input_shape)

    # Initial conv-stem -> MV2 block.
    accX = conv_block(inputs[:,:,:3],filters=filterCount[0])
    gyroX = conv_block(inputs[:,:,3:],filters=filterCount[0])
    accX = mv2Block(accX,expansion_factor,filterCount)
    gyroX = mv2Block(gyroX,expansion_factor,filterCount)
    accX, gyroX  = sensorWiseHART(accX,gyroX, num_blocks=2, projection_dim=projectionDims[0])
    x = tf.concat((accX,gyroX), axis = 2)
    x = layers.Dense(projectionDims[0],activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=projectionDims[0] * expansion_factor, output_channels=filterCount[4], strides=2
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=projectionDims[1])

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=projectionDims[1] * expansion_factor, output_channels=filterCount[5], strides=2
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=projectionDims[2])
    x = conv_block(x, filters=filterCount[6], kernel_size=1, strides=1)
    # Classification head.
    x = layers.GlobalAvgPool1D(name = "GAP")(x)
    
    x = mlp(x, hidden_units=mlp_head_units, dropout_rate=dropout_rate)

    outputs = layers.Dense(activityCount, activation="softmax")(x)
# f.keras.Model(inputs=inputs, outputs=logits)
    return tf.keras.Model(inputs, outputs)

def mobileHART_XXS(input_shape,activityCount,projectionDims = [64,80,96],filterCount = [16//2,16//2,24//2,48//2,64,80,320],expansion_factor=2,mlp_head_units = [1024],dropout_rate = 0.3, droppath_rate = 0.3):
    
    # inputs = keras.Input((segment_size, num_input_channels))
    inputs = layers.Input(shape=input_shape)

    # Initial conv-stem -> MV2 block.
    accX = conv_block(inputs[:,:,:3],filters=filterCount[0])
    gyroX = conv_block(inputs[:,:,3:],filters=filterCount[0])
    accX = mv2Block(accX,expansion_factor,filterCount)
    gyroX = mv2Block(gyroX,expansion_factor,filterCount)
    accX, gyroX  = sensorWiseHART(accX,gyroX, num_blocks=2, projection_dim=projectionDims[0], dropout_rate = dropout_rate, droppath_rate = droppath_rate)
    x = tf.concat((accX,gyroX), axis = 2)
    x = layers.Dense(projectionDims[0],activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=projectionDims[0] * expansion_factor, output_channels=filterCount[4], strides=2
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=projectionDims[1],dropout_rate = dropout_rate, droppath_rate = droppath_rate)

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=projectionDims[1] * expansion_factor, output_channels=filterCount[5], strides=2
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=projectionDims[2],dropout_rate = dropout_rate, droppath_rate = droppath_rate)
    x = conv_block(x, filters=filterCount[6], kernel_size=1, strides=1)
    # Classification head.
    x = layers.GlobalAvgPool1D(name = "GAP")(x)
    
    x = mlp(x, hidden_units=mlp_head_units, dropout_rate=dropout_rate)

    outputs = layers.Dense(activityCount, activation="softmax")(x)
# f.keras.Model(inputs=inputs, outputs=logits)
    return tf.keras.Model(inputs, outputs)




def ispl_inception_decoder(
                   enc_embedding_size,
                   patch_count = 8, 
                   output_shape = (128,6),
                   projection_dim = 256,
                   filters_number = 64,
                   network_depth = 2,
                   use_residual = True,
                   use_bottleneck = True,
                   max_kernel_size = 68,
                   bottleneck_size = 32,
                   regularization_rate = 0.00593,
                   metrics=['accuracy']):
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = layers.Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(layers.Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = layers.Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x
    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = layers.Add()([shortcut_y, out_tensor])
        x = layers.Activation('relu')(x)
        return x

#     input_layer = layers.Input((None, projection_dim))
    
    inputs = layers.Input((patch_count, enc_embedding_size))
    encoded_patches = layers.Dense(projection_dim)(inputs)
    # Build the actual model:
#     input_layer = layers.Input((dim_length, dim_channels))
    x = layers.BatchNormalization()(encoded_patches)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = layers.GlobalAveragePooling1D()(x)
    
    
#     representation = layers.Flatten()(representation)
    pre_final = layers.Dense(units=output_shape[0] * output_shape[1])(gap_layer)
    outputs = layers.Reshape(output_shape)(pre_final)
    return tf.keras.Model(inputs, outputs, name="mae_decoder")

def ispl_inception_encoder(projection_dim,
                   filters_number = 64,
                   network_depth = 5,
                   use_residual = True,
                   use_bottleneck = True,
                   max_kernel_size = 68,
                #    learning_rate = 0.01,
                   bottleneck_size = 32,
                   regularization_rate = 0.00593,
                   metrics=['accuracy']):
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

        # The  channel number is greater than 1
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = layers.Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(layers.Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = layers.Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = layers.Add()([shortcut_y, out_tensor])
        x = layers.Activation('relu')(x)
        return x

    input_layer = layers.Input((None, projection_dim))
    # Build the actual model:
#     input_layer = layers.Input((dim_length, dim_channels))
    x = layers.BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

#     gap_layer = layers.GlobalAveragePooling1D()(x)



    # Final classification layer
#     output_layer = layers.Dense(n_classes, activation="softmax",
#                          kernel_initializer=weightinit, kernel_regularizer=l2(regularization_rate))(gap_layer)

    # Create model and compile
    m = tf.keras.Model(inputs=input_layer, outputs=x)

    # m.compile(loss=out_loss,
    #           optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
    #           metrics=metrics)
    return m


# iSPLInception Model
def ispl_inception(x_shape,
                   n_classes,
                   filters_number = 64,
                   network_depth = 5,
                   use_residual = True,
                   use_bottleneck = True,
                   max_kernel_size = 68,
                #    learning_rate = 0.01,
                   bottleneck_size = 32,
                   regularization_rate = 0.00593,
                   metrics=['accuracy']):
    dim_length = x_shape[0]  # number of samples in a time series
    dim_channels = x_shape[1]  # number of channels
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

        # The  channel number is greater than 1
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = layers.Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(layers.Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = layers.Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = layers.Add()([shortcut_y, out_tensor])
        x = layers.Activation('relu')(x)
        return x

    # Build the actual model:
    input_layer = layers.Input((dim_length, dim_channels))
    x = layers.BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = layers.GlobalAveragePooling1D()(x)



    # Final classification layer
    output_layer = layers.Dense(n_classes, activation="softmax",
                         kernel_initializer=weightinit, kernel_regularizer=l2(regularization_rate))(gap_layer)

    # Create model and compile
    m = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return m

def create_linear_model_from_base_model(base_model, output_shape):
    freezeLayers = len(base_model.layers)
    inputs = base_model.inputs
    x = base_model.output
    outputs = tf.keras.layers.Dense(output_shape,activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=base_model.name + "linear")

    for layer in model.layers[:freezeLayers ]:
        layer.trainable = False
    for layer in model.layers[freezeLayers:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def create_full_classification_model_from_base_model(base_model, output_shape, freeze_fe, model_name="TPN",dropout_rate = 0.3):
    last_freeze_layer = len(base_model.layers)
    intermediate_x = base_model.output
    x = tf.keras.layers.Dense(1024, activation=tf.nn.swish)(intermediate_x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

    for layer in model.layers:
        layer.trainable = True
    
    if(freeze_fe):
        for layer in model.layers[:last_freeze_layer]:
            layer.trainable = False
        print(layer)
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model
