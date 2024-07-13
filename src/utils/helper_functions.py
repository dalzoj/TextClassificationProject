# src/utils/helper_functions.py

import os
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,  # Nivel de registro
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Formato del mensaje
    handlers=[
        logging.FileHandler("log.log"),  # Guardar los logs en un archivo
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)  # Crear un logger

def get_object(objects_path, object_name):
    """
    Load a pickled object from the specified path if it exists.

    Args:
        objects_path (str): The directory path where the object is stored.
        object_name (str): The name of the object file to be loaded.

    Returns:
        object: The loaded object if it exists, otherwise None.
    """
    if os.path.exists(objects_path + object_name):
        with open(objects_path + object_name, 'rb') as archive:
            object_file = pickle.load(archive)
            msg = f' > CREADO: Objeto cargado desde {objects_path}/{object_name}'      
    else:
        object_file = None
        msg = f' > CREADO: Objeto no existente {objects_path}/{object_name}'

    logger.info(msg)    
    return object_file

def save_object(objects_path, object_name, object_file):
    """
    Save an object to a specified path using pickle, overwriting if it already exists.

    Args:
        objects_path (str): The directory path where the object will be saved.
        object_name (str): The name of the object file to be saved.
        object_file (object): The object to be saved.

    Returns:
        None
    """
    if os.path.exists(objects_path + object_name):
        os.remove(objects_path + object_name)
        msg = f' > ELIMINADO: Objeto en {objects_path + object_name}'
    with open(objects_path + object_name, 'wb') as archive:
        pickle.dump(object_file, archive)
        msg = f' > CREADO: Objeto en {objects_path + object_name}'
    logger.info(msg) 

def df_preprocess(df, duplicates = False, data_name = None):
    """
    Preprocess a DataFrame by removing NaN values and optionally duplicates, and resetting the index.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        duplicates (bool): Whether to remove duplicate rows. Default is False.
        data_name (str, optional): Name of the dataset for logging purposes.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    original_shape = df.shape[0]
    if duplicates == True: df = df.drop_duplicates()
    df = df.dropna(how='any')
    df = df.reset_index(drop=True)
    logger.info(f' > PROCESO: Se han eliminado {original_shape-df.shape[0]} de {original_shape} en el preprocesado del archivo {data_name}') 
    return df