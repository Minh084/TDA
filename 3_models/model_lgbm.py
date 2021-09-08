### Developed by Conor Corbin, modified by Minh Nguyen

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
# import tensorflow as tf
import lightgbm as lgb
# import matplotlib.pyplot as plt
import datetime
# import tensorflow_docs as tfdocs
import pytz
import joblib
import os
import argparse
import json
import pdb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from random import randrange
from datetime import timedelta, datetime
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score,plot_precision_recall_curve, plot_roc_curve
# from keras.models import Sequential, model_from_json
# from keras.layers import Dense, Dropout
# from keras.regularizers import l2
# from keras import backend as K


def lightgbm(X_train, y_train, X_valid, y_valid): 

    cv_history = {}

    for lr in [0.01, 0.02, 0.03, 0.05, 0.1]: # 0.3 or 0.5 don't do well
        for num_leaves in [10, 25, 50, 75, 100]: # 10 doesn't do well

            gbm = lgb.LGBMClassifier(objective='binary', n_estimators=1000, learning_rate=lr, num_leaves=num_leaves)

            #For some reason gbm doesn't like pandas dfs... have to use X_train.values - chaning bc we are inputing a sparse matrix
            gbm.fit(X_train, y_train, eval_set= [(X_valid, y_valid)], eval_metric = 'binary', early_stopping_rounds = 10, verbose=False) # True -- too long too much

            y_predict = gbm.predict_proba(X_valid)[:,1]
            cv_roc = roc_auc_score(y_valid, y_predict)
            print("AUC w/ ", lr, "learning rate and ", num_leaves, "num_leaves is ", cv_roc)
            cv_history[cv_roc] = gbm
            
    gbm_opt = cv_history[max(cv_history.keys())]
    y_opt_predict = gbm_opt.predict_proba(X_valid)[:,1]
            
    return y_opt_predict, roc_auc_score(y_valid, y_opt_predict), gbm_opt



def write_params_to_json(clf, output_path, model, TAB):

    json_path = output_path + model + TAB + '_params.json'
    print("Saving hyperparameters to ", json_path)

    if model in ["lasso", "ridge", "elastic_net", "random_forest"]: 
        json.dump(clf.get_params(), open(json_path, 'w'))
    elif model == "lightgbm":
        params = clf.get_params()
        params["n_estimators"] = clf.best_iteration_ #Add in early stopping round 
        json.dump(params, open(json_path, 'w'))
    else:
        #Write the keras model to json 
        model_json = clf.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

            
            
def read_from_json(json_file, model):

    with open(json_file) as file:
        params = json.load(file)

    if model in ["ridge","lasso","elastic_net"]:
        clf = LogisticRegression(**params)
    elif model == "random_forest":
        clf = RandomForestClassifier(**params)
    elif model == "lightgbm":
        clf = lgb.LGBMClassifier(**params)
    elif model == "ffnn":
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        clf = model_from_json(loaded_model_json)
   
    return clf



def retrain_from_model(X_train, y_train, X_test, y_test, clf, model):
    if model in ["ridge","lasso","elastic_net"]:
        
        # #Standardize using log transform
        # log_transformer = FunctionTransformer(np.log1p, validate=True)
        # X_train = log_transformer.transform(X_train)
        # X_test = log_transformer.transform(X_test)
        
        clf.fit(X_train, y_train)
        y_predict = clf.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test, y_predict)
        
    elif model in ["lightgbm", "random_forest"]:
        clf.fit(X_train, y_train)
        y_predict = clf.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test, y_predict)
    else:
        #Standardize using log transform
        log_transformer = FunctionTransformer(np.log1p, validate=True)
        X_train = log_transformer.transform(X_train)
        X_test = log_transformer.transform(X_test)
        
        #Not sure if all of these hyperparameters are saved when loading model -- check this!
        batch_size = 32
        epochs = 100
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        
        clf.compile(loss='binary_crossentropy', 
                          optimizer='adam',
                          metrics=['accuracy', tf.keras.metrics.AUC()])
        
        train_history = model.fit(X_train, y_train, epochs = epochs, 
                      batch_size=batch_size,
                     validation_data=(X_test, y_test), callbacks=[callback])

        y_predict = model.predict(X_test)
        roc = roc_auc_score(y_test, y_predict)
        
    return y_predict, roc



#From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/36031646
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', default=None, type=str, help='ridge/lasso/lightgbm/random_forest/ffnn')
#     parser.add_argument('--feature_types', default='all', type=str, help='Types of features used in model. Options: all,Diagnosis,Imaging,Lab,Meds,Procedures,demo,labs_results_train, vitals_train')
    parser.add_argument('--data_dir', default="", type=str, help='directory with sparce matrices train val test')
    parser.add_argument('--label', default="first_label", type=str, help='the column name for label in cohort')
    parser.add_argument('--output_dir', default="", type=str, help='directory to save outputs')
    parser.add_argument('--val', default=True, type=str2bool, help='True if performing validation (i.e. not using test data)')
    parser.add_argument('--model_file', default = "", type=str, help = "JSON file of validated model hyperparameters")
    args = parser.parse_args()
    model = args.model_class
#     features = args.feature_types
    output_path = args.output_dir
    val_flag = args.val
    model_file = args.model_file

    if model == None:
        print("No model entered.")
        exit()

    if not val_flag and model_file == "":
        print("Performing testing but no trained model from validation entered. Plese enter trained model JSON file.")
        exit()

    TAB = "_validation" if val_flag else "_test"

    if output_path == "":
        output_path = os.getcwd() + "/" + model + TAB + "_results/"
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    print("Using MODEL: ", model)
    
    print("Loading data...")
    
    # when running the .sh file, always do val_flag = 1 first to train the model and select hyperparameters
    # if val_flag is true, meaning we're using train 2015-2017 and validation 2018: hyperparameters tuning
    if val_flag:

        X_train = load_npz(os.path.join(args.data_dir, 'training_examples.npz'))
        y_train = pd.read_csv(os.path.join(args.data_dir, 'training_labels.csv'))
        y_train = y_train[args.label].values

        X_test = load_npz(os.path.join(args.data_dir, 'validation_examples.npz'))
        y_test_df = pd.read_csv(os.path.join(args.data_dir, 'validation_labels.csv'))
        y_test = y_test_df[args.label].values
        
    # when running the .sh file, val_flag = 0 can only be run after results from val_flag = 1
    # if val_flag is false, meaning we're using train_val 2015-2018 and test >= 2019: redo the decile distribution, and do final prediction on test set with the selected best hyperparameters
    else:

        X_train = load_npz(os.path.join(args.data_dir, 'training_and_val_examples.npz'))
        y_train = pd.read_csv(os.path.join(args.data_dir, 'train_and_val_labels.csv'))
        y_train = y_train[args.label].values

        X_test = load_npz(os.path.join(args.data_dir, 'test_examples.npz'))
        y_test_df = pd.read_csv(os.path.join(args.data_dir, 'test_labels.csv'))
        y_test = y_test_df[args.label].values
        
    
    print("Loading data...")
#   X_train, y_train, X_test, y_test, y_test_ids = preprocessing(data, labels,features, validation_time_split, test_time_split, val_flag)

    if val_flag:
        print("Starting training...")
        if model == "ridge":
            predict, roc, clf_opt = ridge(X_train, y_train, X_test, y_test)
        elif model == "lasso":
            predict, roc, clf_opt = lasso(X_train,y_train, X_test, y_test) 
        elif model == "elastic_net":
            predict,roc, clf_opt = elastic_net(X_train, y_train, X_test, y_test)
        elif model == "random_forest":
            predict, roc, clf_opt = random_forest(X_train, y_train, X_test, y_test)
        elif model == "lightgbm":
            predict, roc, clf_opt = lightgbm(X_train, y_train, X_test, y_test)
        elif model == "ffnn":
            predict, roc, clf_opt = ffnn(X_train, y_train, X_test, y_test)
        else:
            print("Model not recognized!")

        print("Optimal cross validated AUROC %s: " % args.model_class, roc)

        #Write optimal hyerparameters to JSON
        write_params_to_json(clf_opt, output_path, model, TAB)

    else:
        print("Fitting trained model.")
        clf = read_from_json(model_file, model)
        predict, roc = retrain_from_model(X_train, y_train, X_test, y_test, clf, model)
        
        print("Test AUROC: ", roc)

    ## Save predictions
    print("Saving predictions to: ", output_path, model, TAB, "_results.csv")
    y_test_df["label"] = y_test
    y_test_df["predictions"] = predict
    y_test_df.to_csv(output_path+model+TAB+"_results.csv")



if __name__ == "__main__":
    main()