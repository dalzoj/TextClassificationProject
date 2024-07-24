# src/features/build_features.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import src.utils.helper_functions as helpers
import src.utils.general_path as general_path

logger = helpers.logger  # Utilizar el logger configurado en helper_functions

import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

features_list = ['tfidf', 'bow']

def build_features(data_name, data, column_name):
    """
    Builds features for the given data using different vectorization methods.

    Parameters:
    data_name (str): Name of the processed dataset.
    data (pandas.DataFrame): Dataset to process.
    column_name (str): Name of the column in the DataFrame that contains the text to vectorize.

    Returns:
    None
    """
    for feature in features_list:
        get_features(
            feature,
            data_name,
            data,
            column_name)

def get_features(feature, data_name, data, column_name):
    """
    Generates and saves a feature object using the specified vectorization method.

    Parameters:
    feature (str): Type of feature to build (e.g., 'tfidf', 'bow').
    data_name (str): Name of the processed dataset.
    data (pandas.DataFrame): Dataset to process.
    column_name (str): Name of the column in the DataFrame that contains the text to vectorize.

    Returns:
    None
    """
    object = helpers.get_object(general_path.OBJECTS_PATH, f'{feature}_{data_name}.pkl')
    if object == None:
        if 'tfidf' in feature:
            object = TfidfVectorizer().fit(data[column_name])
        elif 'bow' in feature:
            object = CountVectorizer().fit(data[column_name])
        helpers.save_object(general_path.OBJECTS_PATH, f'{feature}_{data_name}.pkl', object)
        msg = f' > CREADO: Objeto {feature}_{data_name}.pkl en {general_path.OBJECTS_PATH}{feature}_{data_name}.pkl'
    else:
        msg = f' > AVISO: Objeto {feature}_{data_name}.pkl existente'
    logger.info(msg)

column_name = 'text'

if __name__ == "__main__":
    logger.info(' > INICIO: Script Construcción de Características')
    parser = argparse.ArgumentParser(description='Script construir características.')
    parser.add_argument('--f', type=str, default='n_pre_d_s28817', help='Nombre del conjunto de datos')
    args = parser.parse_args()
    data_name = args.f

    try:
        # Creación de la ruta y cargue del archivo de datos según los argumentos enviados
        data = pd.read_csv(f'{general_path.PREPROCESSED_DATA_PATH}{data_name}.csv')
        data = helpers.df_preprocess(data, True)

        # Construcción de Características
        build_features(
            data_name,
            data,
            column_name)
        
        msg = f'> AVISO: La creación de características se ha efectuado correctamente.'
        logger.info(msg)

    except Exception as e:
        msg = f'> ERROR: Contrucción de características {e}'
        logger.info(msg)