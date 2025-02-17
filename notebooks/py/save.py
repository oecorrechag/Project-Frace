import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict

from mypackage import dir

project = 'france'
data = dir.make_dir(project)
processed = data('processed') 
outputs = data('outputs')

loaded_label_encoders = joblib.load(outputs / 'label_encoders.pkl')

columns_to_labels = ['linea', 'tipo_cliente', 'sexo', 'tipo_identificacion',
                     'estudios', 'canal', 'medio_transaccion', 'tipo_entidad']

orden_columns = ['agencia_origen', 'linea', 'agencia_destino', 'tipo_cliente',
                 'nombre_actividad_economica', 'sexo', 'edad', 'estrato', 'tipo_identificacion',
                 'estudios', 'canal', 'medio_transaccion', 'dias', 'transaccion',
                 'tipo_entidad', 'pred']

nombre_columnas_final = ['Agencia de Origen', 'Linea', 'Agencia de Destino', 'Tipo de Cliente',
                         'Nombre de Actividad Economica', 'Sexo', 'Edad', 'Estrato',
                         'Tipo de Identificacion', 'Estudios', 'Canal', 'Medio de Transaccion', 
                         'Dias de Transaccion', 'Transaccion monetaria', 'Tipo de Entidad', 'Predicción']

# Función auxiliar para merge y renombrar columnas
def merge_and_rename(df_main: pd.DataFrame, df_aux: pd.DataFrame, left_on: str, right_on: str, rename_col: str)-> pd.DataFrame:
    '''
    Toma dos dataframes, merge dataframes por una clave diferente diferente y deja la informacion necesaria
    df_main : pd.DataFrame
        Dataframe principal
    df_aux : pd.DataFrame
        Dataframe secundario
    left_on : str
        Nombre de key en el dataframe principal
    right_on : str
        Nombre de key en el dataframe secundario
    rename_col: str
        Nombre columna posterior al merge    
    '''
    df_main = pd.merge(df_main, df_aux, how='left', left_on=left_on, right_on=right_on)
    df_main = df_main.drop(columns=[left_on, right_on])
    df_main = df_main.rename(columns={rename_col: left_on})
    return df_main

# Función para cargar y guardar los parquets como csv
def comvert_parquet(origin: Path, origin2: Path, 
                    table_name: str, table_name2: str, table_name3: str,
                    columns_to_labels: List[str], 
                    loaded_label_encoders: Dict[str, object], 
                    orden_columns: List[str],  
                    nombre_columnas_final: List[str]) -> pd.DataFrame:
    """
    Convierte archivos Parquet en un DataFrame y realiza transformaciones.

    Parámetros:
    -----------
    origin : str
        Ruta del directorio donde se encuentra el archivo Parquet principal.
    table_name : str
        Nombre del archivo Parquet principal (sin extensión).
    origin2 : str
        Ruta del directorio donde se encuentran los archivos Parquet secundarios.
    table_name2 : str
        Nombre del archivo Parquet de ciudades (sin extensión).
    table_name3 : str
        Nombre del archivo Parquet de actividad económica (sin extensión).
    columns_to_labels : List[str]
        Lista de columnas a transformar usando label encoders.
    loaded_label_encoders : Dict[str, object]
        Diccionario de label encoders previamente cargados.
    orden_columns : List[str]
        Lista de columnas en el orden deseado.
    nombre_columnas_final : List[str]
        Lista de nombres finales para las columnas.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con las transformaciones aplicadas.
    """

    # Cargar DataFrames
    try:
        df = pd.read_parquet(origin / f'{table_name}.parquet.gzip')
        df_ciudades = pd.read_parquet(origin2 / f'{table_name2}.parquet.gzip')
        df_actividad_economica = pd.read_parquet(origin2 / f'{table_name3}.parquet.gzip')
    except FileNotFoundError as e:
        print(f"Error al cargar archivos Parquet: {e}")
        return pd.DataFrame()

    df['agencia_origen'] = df['agencia_origen'].astype(int)
    df['agencia_destino'] = df['agencia_destino'].astype(int)
    df['codigo_actividad'] = df['codigo_actividad'].astype(int)

    df['pred'] = np.where(df['pred'] == -1, 'Atipico', 'Normal')

    # Revertir la transformación
    for col in columns_to_labels:
        if col in loaded_label_encoders:
            le = loaded_label_encoders[col]
            df[col] = le.inverse_transform(df[col])

    conditions = [(df['sexo'] == 'F'), (df['sexo'] == 'M'), (df['sexo'] == '0')]
    values = ['Femenino', 'Masculino', 'Empresa']
    df['sexo'] = np.select(conditions, values)

    # Aplicar merge y renombrar columnas
    df = merge_and_rename(df, df_ciudades, 'agencia_origen', 'AGENCIA', 'Ciudad')
    df = merge_and_rename(df, df_ciudades, 'agencia_destino', 'AGENCIA', 'Ciudad')

    # Merge con actividad económica
    df = pd.merge(df, df_actividad_economica, how='left', on=['codigo_actividad'])
    df = df.drop(columns=['codigo_actividad'])
    df = df.rename(columns={'nombre_actividad_economica_x': 'codigo_actividad'})

    # Reordenar y renombrar columnas finales
    df = df.loc[:, orden_columns]
    df.columns = nombre_columnas_final

    df.to_csv(origin/f'{table_name}.csv', encoding = 'utf-8-sig', index = False)
    print(f'Saved table: {table_name}')

if __name__ == '__main__':
    comvert_parquet(outputs, processed, 'bank', 'df_ciudades', 'df_actividad_economica', columns_to_labels, loaded_label_encoders, orden_columns, nombre_columnas_final)
