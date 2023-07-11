
import utils
import simclr_utitlities
import csv
import model
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

def downStreamPipeline(fineTuneData,fineTuneLabel,valData,valLabel,testData,testLabel,base_save_path,evaluation_dir, initWeightDir,intermediate_model, finetune_epoch = 50,finetune_batch_size = 64, output_shape = 10, feature_extraction_layer = 218, load_ini_weights = True):
    macro_f1_list = []
    intermediate_model.load_weights(base_save_path)
    # No Tune
    tag = "no_tune"

    no_tuneEvaluationResult = simclr_utitlities.evaluate_model_simple(intermediate_model.predict(testData, verbose = 0), testLabel, return_dict=True)

    macro_f1_list.append(no_tuneEvaluationResult['F1 Macro'])
    with open(evaluation_dir +'notune_evaluation_results.csv','w') as f:
        w = csv.writer(f)
        w.writerows(no_tuneEvaluationResult.items())

    # Linear Model

    tag = "linear_eval"
    
    linear_evaluation_model = model.create_linear_model_from_base_model(intermediate_model,
                                                                        output_shape)

    linear_eval_best_model_file_name = evaluation_dir+"linearModelCheckPoint.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=0
    )
    training_history = linear_evaluation_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
#         class_weight=class_weights,
        callbacks=[best_model_callback],
        verbose=2,
        validation_data = (valData,valLabel),
    )

    linear_evaluation_model.load_weights(linear_eval_best_model_file_name)

    linearEvaluationResult = simclr_utitlities.evaluate_model_simple(linear_evaluation_model.predict(testData, verbose = 0), testLabel, return_dict=True)
    macro_f1_list.append(linearEvaluationResult['F1 Macro'])
    with open(evaluation_dir +'linear_evaluation_results.csv','w') as f:
        w = csv.writer(f)
        w.writerows(linearEvaluationResult.items())
    utils.plot_learningCurve(training_history,finetune_epoch,evaluation_dir,'frozen_linear_')

    
  
    # FULL HAR MODEL

    tag = "full_eval"
    full_evaluation_model = model.create_full_classification_model_from_base_model(intermediate_model, output_shape, freeze_fe = True)
    full_eval_best_model_file_name = evaluation_dir+"fullModelCheckPoint.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=0
    )
    
    training_history = full_evaluation_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
#         class_weight=class_weights,
        callbacks=[best_model_callback],
        verbose=2,
        validation_data=(valData,valLabel)
    )
    full_evaluation_model.load_weights(full_eval_best_model_file_name)
    
    # print("Model with lowest validation Acc:")
    fullEvaluationResult = simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(testData, verbose = 0), testLabel, return_dict=True)
    macro_f1_list.append(fullEvaluationResult['F1 Macro'])

    # print(fullEvaluationResult, flush= True)
    with open(evaluation_dir +'full_evaluation_results.csv','w') as f:
        w = csv.writer(f)
        w.writerows(fullEvaluationResult.items())
    utils.plot_learningCurve(training_history,finetune_epoch,evaluation_dir,'frozen_full_')

    # Full HAR Model Unfrozen

    tag = "full_eval_unfrozen"
    
    full_evaluation_model = model.create_full_classification_model_from_base_model(intermediate_model, output_shape, freeze_fe = False)
    full_eval_best_model_file_name = evaluation_dir+"fullModelUnfreezeCheckPoint.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1
    )
    training_history = full_evaluation_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
#         class_weight=class_weights,
        callbacks=[best_model_callback],
        verbose=2,
        validation_data=(valData,valLabel)
    )
    full_evaluation_model.load_weights(full_eval_best_model_file_name)

    # print("Model with lowest validation Acc:")
    fullEvaluationResult = simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(testData, verbose = 0), testLabel, return_dict=True)
    # print(fullEvaluationResult, flush= True)
    macro_f1_list.append(fullEvaluationResult['F1 Macro'])

    with open(evaluation_dir +'unfrozen_full_evaluation_results.csv','w') as f:
        w = csv.writer(f)
        w.writerows(fullEvaluationResult.items())
    utils.plot_learningCurve(training_history,finetune_epoch,evaluation_dir,'unfrozen_full_')

    ### Full HAR Random Model unfrozen

    tag = "full_eval_unfrozen_random"
    if(load_ini_weights):
        intermediate_model.load_weights(initWeightDir)
    full_evaluation_model = model.create_full_classification_model_from_base_model(intermediate_model, output_shape, freeze_fe = False)
    full_eval_best_model_file_name = evaluation_dir+"fullModelUnfreezeRandomCheckPoint.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1
    )
    training_history = full_evaluation_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
#         class_weight=class_weights,
        callbacks=[best_model_callback],
        verbose=2,
        validation_data=(valData,valLabel)
    )
    full_evaluation_model.load_weights(full_eval_best_model_file_name)

    # print("Model with lowest validation Acc:")
    fullEvaluationResult = simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(testData, verbose = 0), testLabel, return_dict=True)
    macro_f1_list.append(fullEvaluationResult['F1 Macro'])

    # print(fullEvaluationResult, flush= True)
    with open(evaluation_dir +'unfrozen_random_full_evaluation_results.csv','w') as f:
        w = csv.writer(f)
        w.writerows(fullEvaluationResult.items())
    utils.plot_learningCurve(training_history,finetune_epoch,evaluation_dir,'unfrozen_rand_full_')
    return macro_f1_list
