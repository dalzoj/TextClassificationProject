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

def build_features(processing_type, data, dataset_size, column_name):
    for feature in features_list:
        get_features(
            f'{feature}_{pt}_s{dataset_size}',
            data,
            processing_type,
            dataset_size,
            column_name)

def get_features(object_name, data, processing_type, dataset_size, column_name):
    object = helpers.get_object(general_path.OBJECTS_PATH, f'{object_name}.pkl')
    if object == None:
        if 'tfidf' in object_name:
            object = TfidfVectorizer().fit(data[column_name])
            object_text = 'tfidf'
        elif 'bow' in object_name:
            object = CountVectorizer().fit(data[column_name])
            object_text = 'bow'
        helpers.save_object(general_path.OBJECTS_PATH, f'{object_name}.pkl', object)
        msg = f' > CREADO: Objeto {object_text}_{processing_type}_s{dataset_size} en {general_path.OBJECTS_PATH}{object_text}_{processing_type}_s{dataset_size}.pkl'
    else:
        msg = f' > AVISO: Objeto {object_name}.pkl existente'
    logger.info(msg)

column_name = 'text'

if __name__ == "__main__":
    logger.info(' > INICIO: Script Construcción de Características')
    parser = argparse.ArgumentParser(description='Script construir características.')
    parser.add_argument('--pt', type=str, default='normal', help='Tipo de procesamiento: "normal" o "spellchecker"')
    parser.add_argument('--s', type=str, default='28817', help='Tamaño del conjunto de datos')
    args = parser.parse_args()
    processing_type = args.pt
    dataset_size = args.s

    # Nomenclatura de tipo de preprocesado
    if processing_type == 'normal': pt = 'n'
    elif processing_type == 'spellchecker': pt = 'sc'

    # Creación de la ruta y cargue del archivo de datos según los argumentos enviados
    data_path = f'{general_path.PREPROCESSED_DATA_PATH}{pt}_pre_d_s{dataset_size}.csv'
    data = pd.read_csv(data_path)
    data = helpers.df_preprocess(data, True)

    try:
        build_features(
            pt,
            data,
            dataset_size,
            column_name)
        msg = f'> AVISO: La creación de características se ha efectuado correctamente.'
        logger.info(msg)
    except Exception as e:
        msg = f'> ERROR: Contrucción de características {e}'
        logger.info(msg)