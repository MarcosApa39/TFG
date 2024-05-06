import pandas as pd

# Cargar el archivo Excel
archivo_excel = 'BaseNaked.xlsx'
df = pd.read_excel(archivo_excel)

df = df.dropna()

# Eliminar filas que no tienen el valor 'Venta' en la columna 'Línea'
df = df[df['Línea'] == 'Venta']
df = df[df['Budget/Real'] == 'Real']

# Eliminar filas que tienen el valor 'Venta' en la columna 'Línea' y el valor 0 en la columna 'Importe'
df = df[~((df['Línea'] == 'Venta') & (df['Importe'] == 0))]

# Ordenar el DataFrame por la columna 'Año', 'Restaurante' y 'Mes'
df = df.sort_values(by=['Año', 'Restaurante', 'Mes'])

# Guardar el DataFrame modificado en un nuevo archivo Excel
df.to_excel('BaseNaked_Modified3.xlsx', index=False)

# Mostrar el DataFrame resultante
print(df)
