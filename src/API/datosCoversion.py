import pandas as pd               # Para la manipulación y análisis de datos

directorio = 'E:\Escuela 9\Mineria de datos\Proyecto\demo-app\src\API\datosPrueba.csv'
datos = pd.read_csv(directorio)
datos.to_json("E:\Escuela 9\Mineria de datos\Proyecto\demo-app\src\API\data.json", orient = "records")