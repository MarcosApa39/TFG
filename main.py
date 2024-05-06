import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Leer los archivos Excel
archivo_ventas = 'Ventas_Naked_CCAA.xlsx'
archivo_base_final = 'BaseFinalCCAA.xlsx'

df_ventas = pd.read_excel(archivo_ventas, index_col='CCAA')
df_base_final = pd.read_excel(archivo_base_final, index_col='CCAA')

# Combinar los DataFrames en función de la columna 'CCAA'
df_combinado = pd.merge(df_base_final, df_ventas, left_index=True, right_index=True, how='inner')

# Resetear el índice y convertirlo en una columna separada
df_combinado.reset_index(inplace=True)




# Guardar el DataFrame combinado en un nuevo archivo Excel
df_combinado.to_excel("Naked_CCAA_Merged.xlsx", index=False)



# Seleccionar solo columnas numéricas
numeric_columns = df_combinado.select_dtypes(include='number')

# Definir una función para formatear los valores en millones de euros
def millones_formatter(x, _):
    return f'€{int(x/1e6):,}M'

# Definir una paleta de colores personalizada
colores = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]

# Gráfico de ventas con mejoras visuales
plt.figure(figsize=(12, 8))

# Utilizar la paleta de colores y agregar bordes más oscuros a las barras
sns.barplot(x='CCAA', y='Importe', data=df_combinado, palette=colores, edgecolor='black')

# Formatear los valores en millones de euros en el eje y
plt.gca().yaxis.set_major_formatter(FuncFormatter(millones_formatter))

# Agregar títulos y etiquetas
plt.title('Gráfico de ventas Naked&Sated', fontsize=16)
plt.xlabel('Comunidades Autónomas (CCAA)', fontsize=14)
plt.ylabel('Ventas (Millones de €)', fontsize=14)

# Añadir decoraciones adicionales si es necesario (por ejemplo, grid)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Rotar las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()




# Establecer la primera columna como índice
df_combinado.set_index(df_combinado.columns[0], inplace=True)

correlation_matrix = df_combinado.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de correlación de todas las variables')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()

