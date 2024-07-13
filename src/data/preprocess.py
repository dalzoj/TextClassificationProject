import re
import os
import argparse
import pandas as pd
import src.utils.general_path as general_path
import src.utils.helper_functions as helpers
from unidecode import unidecode
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from sklearn.preprocessing import LabelEncoder

logger = helpers.logger  # Utilizar el logger configurado en helper_functions

# Archivos Base
raw_analytic_name = 'analytic_data.csv'
raw_correct_name = 'correct_register.xlsx'

# Expresión regular para texto
letters_pat = re.compile(r'[^a-z ]')
# Expresión regular para múltiples espacios
spaces_pat = re.compile(r' +')
# Expresión regular para puntos y comas
points_pat = re.compile(r'[,.-]')
# Crear el corrector ortográfico en español
spell = SpellChecker(language='es')
# Crear las Stop Words
stop_words = stopwords.words("spanish")

def text_processing(text):
    """
    Process the given text by normalizing, removing unwanted characters, punctuation, and stopwords.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The processed text with unwanted characters, punctuation, and stopwords removed.
    """
    processed_text = unidecode(text.lower())
    processed_text = re.sub(letters_pat, " ", processed_text)
    processed_text = re.sub(points_pat, " ", processed_text)
    processed_text = re.sub(spaces_pat, " ", processed_text)
    processed_text = processed_text.strip()
    tokens = processed_text.split()
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

def text_spellchecker_processing(text):
    """
    Process the given text by normalizing, removing unwanted characters, punctuation, 
    correcting spelling errors, and removing stopwords.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The processed text with spelling corrections and unwanted characters, punctuation, and stopwords removed.
    """
    processed_text = unidecode(text.lower())
    processed_text = re.sub(letters_pat, " ", processed_text)
    processed_text = re.sub(points_pat, " ", processed_text)
    processed_text = re.sub(spaces_pat, " ", processed_text)
    processed_text = processed_text.strip()
    tokens = processed_text.split()
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [spell.correction(token) if spell.correction(token) else token for token in tokens]
    print(unidecode(" ".join(tokens)))
    return unidecode(" ".join(tokens))

def preprocess(data, column_name, processing_type):
    """
    Preprocess the specified column in the given DataFrame using the specified processing type.

    Args:
        data (pd.DataFrame): The input DataFrame containing the text data.
        column_name (str): The name of the column to be processed.
        processing_type (str): The type of processing to apply ('normal' or 'spellchecker').

    Returns:
        pd.DataFrame: The DataFrame with the specified column processed and preprocessed.
    """
    if processing_type == 'normal':
        data[column_name] = data[column_name].apply(text_processing)
        data = helpers.df_preprocess(data, True)
    elif processing_type == 'spellchecker':
        data[column_name] = data[column_name].apply(text_spellchecker_processing)
        data = helpers.df_preprocess(data, True)
    return data

if __name__ == "__main__":
    logger.info(' > INICIO: Script Preprocesamiento de Datos')
    parser = argparse.ArgumentParser(description='Script para procesar datos.')
    parser.add_argument('--processing_type', type=str, default='normal', help='Tipo de procesamiento: "normal" o "spellchecker"')
    args = parser.parse_args()
    processing_type = args.processing_type

    # Carga de datos
    logger.info(' > Carga de datos')
    raw_analytic_df = pd.read_csv(general_path.RAW_DATA_PATH + raw_analytic_name)
    raw_correct_df = pd.read_excel(general_path.RAW_DATA_PATH + raw_correct_name)

    logger.info(' > Adecuacion de datos')
    # Correción de datos
    correctly_processed_df = raw_correct_df.rename(
    columns = {
        'hechos_sample': 'text',
        'Clasificación': 'label'
        }
    )
    correctly_processed_df = correctly_processed_df[['label','text']]
    correctly_processed_df = correctly_processed_df.replace(
        to_replace = {
            'QUEJA': 'QJA',
            'SOLICITUD': 'SOL',
            'ASESORIA': 'ASE'
            }
        )
    correctly_processed_df = helpers.df_preprocess(correctly_processed_df, True, raw_correct_name)
    processed_analytical_df = helpers.df_preprocess(raw_analytic_df, True, raw_analytic_name)
    preprocess_df = pd.concat(
        [correctly_processed_df, processed_analytical_df]
        )
    preprocess_df = helpers.df_preprocess(preprocess_df, True, 'resultado')

    # Extensión de Stopwords
    logger.info(' > CARGA: Stopwords')
    aditional_stopwords_list = general_path.EXTENDS_DATA_PATH + 'words_to_ignore.txt'
    stop_words.extend(aditional_stopwords_list)

    # Preprocesado de datos
    logger.info(f' > PROCESO: Preprocesado de datos tipo {processing_type}')
    preprocess_df = preprocess(preprocess_df, 'text', processing_type)

    # Guarde de información
    preprocess_df.to_csv(
        f'{general_path.PROCESSED_DATA_PATH}{processing_type}_data_s{preprocess_df.shape[0]}.xlsx',
        index = False
        )
    msg = f' > CREADO: Archivo {processing_type}_data_s{preprocess_df.shape[0]}.xlsx en {general_path.PROCESSED_DATA_PATH}{processing_type}_data_s{preprocess_df.shape[0]}.xlsx'
    logger.info(msg)
    
    # Codificación de categorías
    le = helpers.get_object(general_path.OBJECTS_PATH, f'label_encoder_s{preprocess_df.shape[0]}.pkl')
    if le == None:
        le = LabelEncoder().fit(preprocess_df['label'])
        helpers.save_object(general_path.OBJECTS_PATH, f'label_encoder_s{preprocess_df.shape[0]}.pkl', le)
    else:
        msg = f' > CARGADO: {general_path.OBJECTS_PATH}label_encoder_s{preprocess_df.shape[0]}.pkl'


    # Transformación de las categorías
    y_encoder = le.fit_transform(preprocess_df['label'])
    preprocess_df['label'] = y_encoder

    preprocess_df.to_csv(
        f'{general_path.PROCESSED_DATA_PATH}{processing_type}_processed_data_s{preprocess_df.shape[0]}.csv',
        index = False
        )
    msg = f' > CREADO: Archivo {processing_type}_processed_data_s{preprocess_df.shape[0]}.csv en {general_path.PROCESSED_DATA_PATH}{processing_type}_processed_data_s{preprocess_df.shape[0]}.csv'
    logger.info(msg)