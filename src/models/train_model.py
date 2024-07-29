# src/models/train_model.py
import os
import sys

sys.path.append(
   os.path.abspath(
      os.path.join(
         os.path.dirname(__file__),
         '..',
         '..'
         )
      )
   )

import src.utils.helper_functions as helpers
import src.utils.general_path as general_path

import pandas as pd
import numpy as np
import pprint as pp
import joblib
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier

logger = helpers.logger 

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

def save_model(model_output_name, model_result, metric, model):
    model_name = f"{model_output_name}_{metric}_tn_{str(model_result['train'][metric])}_tt_{str(model_result['test'][metric])}"
    joblib.dump(model, f'{general_path.SAVE_MODELS_PATH}{model_name}.joblib')
    logger.info(f' > > > GUARDADO: Objeto del modelo {model_name}.joblib en {general_path.SAVE_MODELS_PATH}{model_name}.joblib')


def result_record(model_name, file_name, feature_name, best_params, models_result):
    with open(f'{general_path.MODELS_PATH}model_log.txt', 'a') as f:
        f.write(f'Nombre Modelo: {model_name} \n')
        f.write(f'Conjunto de Datos: \n')
        f.write(f'   {file_name} \n')
        f.write(f'Representación Textual: \n')
        f.write(f'   {feature_name} \n')
        f.write(f'Resultados: \n')
        f.write(f'  -> Entrenamiento:\n')
        for key, value in models_result['train'].items():
            f.write(f'       {key}  {value} \n')
        f.write(f'  -> Testeo: \n')
        for key, value in models_result['test'].items():
            f.write(f'       {key}  {value} \n')
        f.write(f'Mejores Hiperparámetros: {model_name} \n')
        f.write('\n'.join(f'   {key}: {value}' for key, value in best_params.items()))
        f.write(' \n \n')
    logger.info(f' > > GUARDADO: Registro del modelo {model_name}')
   
def apply_grid_search(model_key, model_value, X_train, y_train, X_test, y_test, metric='recall'):        
    grid_search = GridSearchCV(
        estimator=model_value[0],
        param_grid=model_value[1],
        scoring=scoring,
        refit=metric)

    grid_search.fit(X_train, y_train)

    y_pred = grid_search.best_estimator_.predict(X_test)
    
    model_result = {
        'train' : {
            'accuracy' :  str(round(grid_search.cv_results_['mean_test_accuracy'][0],2)),
            'precision' :  str(round(grid_search.cv_results_['mean_test_precision'][0],2)),
            'recall' :  str(round(grid_search.cv_results_['mean_test_recall'][0],2)),
            'f1' :  str(round(grid_search.cv_results_['mean_test_f1'][0],2))
        },
        'test' :{
            'accuracy' :  str(round(accuracy_score(y_test,y_pred),2)),
            'precision' :  str(round(precision_score(y_test,y_pred, average='weighted'),2)),
            'recall' :  str(round(recall_score(y_test,y_pred, average='weighted'),2)),
            'f1' :  str(round(f1_score(y_test,y_pred, average='weighted'),2))
        }

    }

    # Grabado de resultados
    result_record(model_key, file_name, feature_name, grid_search.best_params_, model_result)

    # Guardado de modelos
    model_output_name = f'{model_key}_{feature_name}_{file_name}'
    save_model(model_output_name, model_result, metric, grid_search.best_estimator_)

    logger.info(f' > > > FINALIZADO: Entrenamiento modelo {model_key}')    
        

def train_model(data, file_name, column_name, model_name, model_dict, feature, feature_name, test_size=0.20, random_state = 42, metric = 'recall'):
    X_train, X_test, y_train, y_test = train_test_split(
        data[column_name[0]], data[column_name[1]],
        test_size = test_size,
        random_state = random_state,
        stratify = data[column_name[1]]
        )

    X_train_transform = feature.transform(X_train)
    X_test_transform = feature.transform(X_test)

    if model_name == 'all':
        logger.info(f' > > INICIO: Entrenamiento de todos los modelos')
        for model_key, model_value in model_dict.items():
            logger.info(f' > > > INICIO: Entrenamiento del modelo {model_key}')
            apply_grid_search(model_key, model_value, X_train_transform, y_train, X_test_transform, y_test)  
    else:
        logger.info(f' > > INICIO: Entrenamiento de modelo invidivual')
        for model_key, model_value in model_dict.items():
            if model_key == model_name:      
                logger.info(f' > > > INICIO: Entrenamiento modelo {model_name}')
                apply_grid_search(model_key, model_value, X_train_transform, y_train, X_test_transform, y_test)
    logger.info(f' > > FINALIZADO: Entrenamiento') 
   

def get_model_dict(param_grid):
    model_dict = {
       'KNeighborsClassifier': [KNeighborsClassifier(), param_grid['KNeighborsClassifier']],
       'RandomForestClassifier': [RandomForestClassifier(), param_grid['RandomForestClassifier']],
       'XGBClassifier': [XGBClassifier(eval_metric='logloss'), param_grid['XGBClassifier']],
       'DecisionTreeClassifier': [DecisionTreeClassifier(), param_grid['DecisionTreeClassifier']],
       'SVC': [SVC(), param_grid['SVC']],
       'GaussianNB': [GaussianNB(), param_grid['GaussianNB']],
       'MLPClassifier': [MLPClassifier(), param_grid['MLPClassifier']],
    }
    return model_dict

def load_data(file_name):
    data = pd.read_csv(general_path.PREPROCESSED_DATA_PATH + f'{file_name}.csv')
    return helpers.df_preprocess(data, True)
   
def load_feature_object(feature_name, file_name):
   return helpers.get_object(general_path.OBJECTS_PATH,f'{feature_name}_{file_name}.pkl')

if __name__ == "__main__":
    logger.info(' > INICIO: Script Entrenamiento de Modelos')
    parser = argparse.ArgumentParser(description='Script Entrenamiento de Modelos')
    parser.add_argument('--tr', type=str, default='tfidf', help='Tipo de Representación Textual')
    parser.add_argument('--f', type=str, default='n_pre_d_s28817', help='Nombre del conjunto de datos')
    parser.add_argument('--m', type=str, default='RandomForestClassifier', help='Nombre del modelo para entrenamiento.')
    parser.add_argument('--pg', type=str, default='classification_param_grid_small', help='Nombre de archivo YAML de los hiperparámetros de modelos')
    args = parser.parse_args()
    feature_name = args.tr
    file_name = args.f 
    model_name = args.m
    param_grid_name = args.pg

    # Carga de características
    feature = load_feature_object(feature_name, file_name)
    print(feature)
    logger.info(f' > CARGADO: Se ha cargado el objeto {feature_name}_{file_name}.pkl') 
    
    # Carga de datos
    data = load_data(file_name)
    logger.info(f' > CARGADO: Se ha cargado el archivo {file_name}.csv') 

    # Carga de hiperparámetros de modelos
    param_grid = helpers.load_yaml_file(general_path.MODELS_PATH, param_grid_name)
    logger.info(f' > CARGADO: Hiperparámetros de modelos de {param_grid_name}.yaml')

    # Creación de diccionario con la información de los modelos
    model_dict = get_model_dict(param_grid)
    logger.info(f' > CREADO: Diccionario con información de modelos')

    logger.info(f' > INICIADO: Proceso de Entrenamiento de Modelos')
    train_model(data, file_name, ['text','label'], model_name, model_dict, feature, feature_name)
    logger.info(f' > TERMINADO: Proceso de Entrenamiento de Modelos')