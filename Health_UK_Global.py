import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

# Load the Excel file
df_Regions = pd.read_excel('Health_Index_Score_UK.xlsx')

# Select specific columns and rename them
selected_columns = ['Area Name', 'Healthy People Domain', 'Life satisfaction [Pe4]',
                    'Physical health conditions [Pe]', 'Diabetes [Pe5]', 'Healthy eating [L1]',
                    'Overweight and obesity in adults [L3]',
                    'Physical activity [L1]', 'Economic and working conditions [Pl]', 'Mental health [Pe]',
                    'Life expectancy [Pe3]', 'Unemployment [Pl4]', 'Alcohol misuse [L1]']

df_Regions = df_Regions[selected_columns]
new_headers_df = ['Region', 'Healthy People Domain', 'Life satisfaction',
                   'Physical health conditions', 'Diabetes', 'Healthy eating',
                   'Overweight and obesity in adults',
                   'Physical activity', 'Economic condition', 'Mental health',
                   'Life expectancy', 'Unemployment', 'Alcohol']
df_Regions.columns = new_headers_df

# Eliminar valores nulos
df_Regions.dropna(inplace=True)




# Seleccionar solo columnas numéricas
numeric_columns = df_Regions.select_dtypes(include='number')

# Gráfico de dispersión de 'Physical Activity'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Physical activity', y='Healthy eating', data=df_Regions)
plt.title('Dispersion graph between Healthy Eating and Physical Activity')
plt.xlabel('Physical activity')
plt.ylabel('Healthy Eating')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()

# Gráfico de dispersión de 'Physical Activity'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Healthy eating', y='Mental health', data=df_Regions)
plt.title('Dispersion graph between Healthy Eating and Mental health')
plt.xlabel('Healthy Eating')
plt.ylabel('Mental Health')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()

# Gráfico de dispersión de 'Physical Activity'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Economic condition', y='Healthy eating', data=df_Regions)
plt.title('Dispersion graph between Economic condition and Healthy Eating')
plt.xlabel('Economic condition')
plt.ylabel('Healthy eating')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x='Healthy eating', y='Life expectancy', data=df_Regions)
plt.title('Dispersion graph between Life expectancy and Healthy Eating')
plt.xlabel('Healthy eating')
plt.ylabel('Life expectancy')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()

# Matriz de correlación

# Eliminar la columna 'Region' temporalmente para calcular límites de outliers
numeric_columns = df_Regions.drop('Region', axis=1)

# Calcular los límites superior e inferior para cada variable numérica
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Aplicar el filtro para eliminar outliers
outlier_filter = ((numeric_columns >= lower_limit) & (numeric_columns <= upper_limit)).all(axis=1)
df_no_outliers = df_Regions[outlier_filter]

# Gráfico de caja y bigotes después de eliminar outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_no_outliers.drop('Region', axis=1))
plt.title('Gráficos de Caja y Bigotes sin Outliers', fontsize=16)
plt.xlabel('Variables', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x para mejorar la legibilidad
plt.tight_layout()
plt.show()


# Establecer la primera columna como índice
df_Regions.set_index(df_Regions.columns[0], inplace=True)

correlation_matrix = df_Regions.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de correlación de todas las variables')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()


# Calcular medidas de tendencia central para cada variable numérica
for column in numeric_columns.columns:
    media = numeric_columns[column].mean()
    mediana = numeric_columns[column].median()
    moda = numeric_columns[column].mode().iloc[0]

    print(f"Medidas de tendencia central para '{column}':")
    print(f"   Media: {media:.2f}")
    print(f"   Mediana: {mediana:.2f}")
    print(f"   Moda: {moda:.2f}")
    print()

# Calcular medidas de dispersión para cada variable numérica
for column in numeric_columns.columns:
    rango = numeric_columns[column].max() - numeric_columns[column].min()
    varianza = numeric_columns[column].var()
    desviacion_tipica = numeric_columns[column].std()

    print(f"Medidas de dispersión para '{column}':")
    print(f"   Rango: {rango:.2f}")
    print(f"   Varianza: {varianza:.2f}")
    print(f"   Desviación Típica: {desviacion_tipica:.2f}")
    print()

# Calcular medidas de frecuencia para cada variable numérica
for column in numeric_columns.columns:
    # Crear intervalos (por ejemplo, 5 intervalos)
    intervalos = pd.cut(numeric_columns[column], bins=5, precision=2)

    # Contar la frecuencia de observaciones en cada intervalo
    frecuencia_intervalos = intervalos.value_counts().sort_index()

    print(f"Medidas de frecuencia para '{column}':")
    print(frecuencia_intervalos)
    print()
