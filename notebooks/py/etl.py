import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mypackage import dir

# Environment variables
project = 'france'
data = dir.make_dir(project) 
raw = data('raw')
processed = data('processed')
outputs = data('outputs')

columns_to_labels = ['linea', 'tipo_cliente', 'sexo', 'tipo_identificacion',
                     'estudios', 'canal', 'medio_transaccion', 'tipo_entidad']
label_encoders = {}

# Función para cargar datos
def cargar_datos(table_name: str) -> pd.DataFrame:
    df = pd.read_csv(raw / f'{table_name}.csv', sep = ',', decimal = '.', header = 0, encoding = 'latin1')
    print(f'Loaded table: {table_name}')
    return df


# Función para transformar los datos clientes naturales
def transformar_datos_naturales(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df['NUMEROCUEN'] = df['NUMEROCUEN'].astype(np.int64).astype(str)
    df['FECHAACTUA'] = pd.to_datetime(df['FECHAACTUA'], origin='1899-12-30', unit='D')
    df['FECHAACTUA'] = pd.to_datetime(df['FECHAACTUA'], dayfirst='true', errors = 'coerce')
    df['tipo'] = 'Narutales'

    df = df[df['NATURALEZA'] == 'Consig']
    df = df.loc[:,['AGENCIA', 'CODLINEA', 'AGENCIATRA',
                   'CLIENTEtipo', 'CODACTIVID', 'SEXO', 'EDAD', 'ESTRATO', 'TIPOIDENTI', 'ESTUDIOS',
                   'NUMEROCUEN', 'CANAL', 'MEDIOTRANS', 'FECHAACTUA', 'INGRESOS1', 
                   'tipo'
                   ]]

    df = df.drop_duplicates()
    return df


# Función para transformar los datos clientes juridicos
def transformar_datos_juridicos(df: pd.DataFrame, df_ciudades: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df['FECHAACTUA'] = pd.to_datetime(df['FECHAACTUA'], dayfirst='true', errors = 'coerce')
    df['NUMEROCUEN'] = df['NUMEROCUEN'].astype(np.int64).astype(str)
    df['tipo'] = 'Juridicas'

    df = df[df['NATURALEZA'] == 'Consig']
    df['ESTUDIOS'] = np.where(df['ESTUDIOS'] == '0', 'O', df['ESTUDIOS'])
    df.loc[df['INGRESOS1'] < 0, 'INGRESOS1'] = 0
    df.loc[df['EDAD'].isin([18, 27, 35]), 'EDAD'] = 118

    df = df.loc[:,['Ciudad', 'CODLINEA', 'AGENCIATRA',
                   'CLIENTEtipo', 'CODACTIVID', 'SEXO', 'EDAD', 'ESTRATO', 'TIPOIDENTI', 'ESTUDIOS',
                   'NUMEROCUEN', 'CANAL', 'MEDIOTRANS', 'FECHAACTUA', 'INGRESOS1',
                   'tipo', 
                   ]]

    df = df.drop_duplicates()

    print(df.shape)
    df = pd.merge(df, df_ciudades, how='left', on=['Ciudad'])    
    print(df.shape)

    df = df.loc[:,['AGENCIA', 'CODLINEA', 'AGENCIATRA',
                   'CLIENTEtipo', 'CODACTIVID', 'SEXO', 'EDAD', 'ESTRATO', 'TIPOIDENTI', 'ESTUDIOS',
                   'NUMEROCUEN', 'CANAL', 'MEDIOTRANS', 'FECHAACTUA', 'INGRESOS1',
                   'tipo',
                   ]]
    return df


# Función para crear el dataset de entrenamiento
def crear_dataset(df_naturales: pd.DataFrame, df_juridicas: pd.DataFrame) -> pd.DataFrame:

    df = pd.concat([df_naturales, df_juridicas], ignore_index=True)

    df = df.sort_values(by=['FECHAACTUA'], ascending=True)
    fecha_actual = datetime.now()
    df['dias'] = (fecha_actual - df['FECHAACTUA']).dt.days

    for col in ['AGENCIA', 'AGENCIATRA', 'CODACTIVID', 'ESTRATO']:
        df[col] = df[col].astype(str)

    df.columns = ['agencia_origen', 'linea', 'agencia_destino',
                'tipo_cliente', 'codigo_actividad', 'sexo', 'edad', 'estrato', 'tipo_identificacion', 'estudios',
                'numero_cuenta', 'canal', 'medio_transaccion', 'fecha_actual',  'transaccion',
                'tipo_entidad', 'dias'
                ]

    df = df.loc[:,['agencia_origen', 'linea', 'agencia_destino',
                'tipo_cliente', 'codigo_actividad', 'sexo', 'edad', 'estrato', 'tipo_identificacion', 'estudios',
                'numero_cuenta', 'canal', 'medio_transaccion', 'dias',  'transaccion',
                'tipo_entidad', 
                ]]

    return df


def almacenar_en_db(df: pd.DataFrame, table_name: str) -> None:
    df.to_parquet(processed/f'{table_name}.parquet.gzip', compression='gzip')
    print(f'Saved table: {table_name}')


# Función para cargar los datos en la base de datos
def cargar_en_db(df: pd.DataFrame, table_name: str) -> None:
    df.to_parquet(processed/f'{table_name}.parquet.gzip', compression='gzip')
    print(f'Saved table: {table_name}')


if __name__ == '__main__':

    # ETL Datos naturales
    df_naturales = cargar_datos('Naturales')
    # Crear una copia para sacar datos
    df_ciudades = df_naturales.copy()
    df_actividad_economica = df_naturales.copy()
    df_naturales = transformar_datos_naturales(df_naturales)

    # ETL ciudades
    df_ciudades2 = df_ciudades.loc[:,['AGENCIA', 'Ciudad']]
    df_ciudades3 = df_ciudades.loc[:,['AGENCIATRA', 'CiudadTransac']]
    df_ciudades3.rename(columns={'AGENCIATRA':'AGENCIA', 'CiudadTransac':'Ciudad'}, inplace=True)
    df_ciudades = pd.concat([df_ciudades2, df_ciudades3], ignore_index=True).drop_duplicates()
    df_ciudades = df_ciudades.sort_values(by=['AGENCIA'], ascending=True).reset_index(drop=True)
    almacenar_en_db(df_ciudades, 'df_ciudades')

    # ETL actividad economica
    df_actividad_economica = df_actividad_economica.loc[:,['CODACTIVID', 'NOMBREACTI']]
    df_actividad_economica = df_actividad_economica.drop_duplicates()
    df_actividad_economica['NOMBREACTI'] = df_actividad_economica['NOMBREACTI'].str.rstrip('.')
    df_actividad_economica = df_actividad_economica.sort_values(by=['CODACTIVID'], ascending=True)
    df_actividad_economica.columns = ['codigo_actividad', 'nombre_actividad_economica']
    df_actividad_economica['nombre_actividad_economica'] = df_actividad_economica['nombre_actividad_economica'].str.capitalize()
    almacenar_en_db(df_actividad_economica, 'df_actividad_economica')

    # ET Datos juridicas
    df_juridicas = cargar_datos('Juridicas')
    df_juridicas = transformar_datos_juridicos(df_juridicas, df_ciudades)

    # Create dataset
    df = crear_dataset(df_naturales, df_juridicas)
    df.drop(['numero_cuenta'], axis=1, inplace=True)
    df = shuffle(df)

    # Aplicar LabelEncoder a cada columna categórica
    for col in columns_to_labels:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    joblib.dump(label_encoders, f'{outputs}/label_encoders.pkl')

    train_df, temp_df = train_test_split(df, test_size=0.9, random_state=42)

    almacenar_en_db(train_df, 'dataset_train')
    almacenar_en_db(temp_df, 'dataset_val')
