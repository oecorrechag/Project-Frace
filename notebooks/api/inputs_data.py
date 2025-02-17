# 1. Library imports
from dataclasses import dataclass

# 2. Class for models.
@dataclass
class Inputs:
    agencia_origen: int = 1
    linea: int = 2
    agencia_destino: int = 1
    tipo_cliente: int = 1
    codigo_actividad: int = 10
    sexo: int = 1
    edad: int = 44
    estrato: int = 3
    tipo_identificacion: int = 1
    estudios: int = 1
    canal: int = 1
    medio_transaccion: int = 1
    dias: int = 1
    transaccion: int = 1
    tipo_entidad: int = 1
    