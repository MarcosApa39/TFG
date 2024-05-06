import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Aquí, inicializamos el dataframe final con la primera columna de Comunidades Autónomas:
# Lista de valores para la columna 'CCAA'
ccaa_values = ['Andalucía', 'Aragón', 'Asturias, Principado de', 'Balears, Illes', 'Canarias',
               'Cantabria', 'Castilla y León', 'Castilla-La Mancha', 'Cataluña', 'Comunitat Valenciana',
               'Extremadura', 'Galicia', 'Madrid, Comunidad de', 'Murcia, Región de',
               'Navarra, Comunidad Foral de', 'País Vasco', 'Rioja, La']

# Crear un DataFrame con la columna 'CCAA'
df_ccaa = pd.DataFrame({'CCAA': ccaa_values})

""" 1. Modificamos el Excel de Salud para crear un indicador del estado físico en cada Comunidad Autónoma:"""
# Cargar el archivo 'CCAA_Salud.xlsx'
salud_file_path = 'CCAA_Salud.xlsx'

# Leer el archivo excluyendo la segunda fila de encabezados
df_salud = pd.read_excel(salud_file_path, header=[0], skiprows=[1])

# Normalizar los valores dividiendo entre los valores de la columna 'Total'
df_salud.iloc[:, 1:] = df_salud.iloc[:, 1:].div(df_salud['Total'].values, axis=0)

# Eliminar la primera fila que no corresponde a los encabezados
df_salud = df_salud.iloc[:-3]

# Crear un nuevo DataFrame con las columnas 'CCAA' y 'Health Score'
df_salud_final = pd.DataFrame({'CCAA': df_salud.iloc[:, 0]})

# Definir la fórmula para la columna 'Health Score'
formula2 = (df_salud['Muy bueno'] * 3 +
           df_salud['Bueno'] * 2 +
           df_salud['Malo'] * (-2) +
           df_salud['Muy malo'] * (-3)) * 100

# Agregar la columna 'Healt Score' al nuevo DataFrame
df_salud_final['CCAA'] = df_ccaa['CCAA']
df_salud_final['Health Score'] = formula2

# Añade la columna 'Renta_2022' al DataFrame final
df_ccaa['Healthy People Domain'] = df_salud_final['Health Score']



"""2. Modificamos el Excel de Felicidad para elaborar un índice que llamaremos Life Satisfaction"""

# Cargar el archivo 'CCAA_Felicidad.xlsx'
file_happiness = 'CCAA_Felicidad.xlsx'

# Leer el archivo excluyendo la segunda fila de encabezados
df_Felicidad = pd.read_excel(file_happiness, header=[0, 1])
df_Felicidad = df_Felicidad.iloc[1:-2]
# Seleccionar la columna 'Muy a menudo' para el año 2022
felicidad_2022 = df_Felicidad['Muy a menudo', '2022']

# Crear un nuevo DataFrame con las columnas 'CCAA' y 'Felicidad 2022'
df_felicidad_final = pd.DataFrame({'CCAA': df_Felicidad.index.get_level_values(0), 'Life Satisfaction': df_Felicidad['Muy a menudo', '2022'].values})

# Añadir la columna 'Life Satisfaction' al DataFrame final
df_ccaa['Life Satisfaction'] = df_felicidad_final['Life Satisfaction']

"""3. Modificamos el Excel de enfermedades para elaborar un índice de las condiciones físicas de las personas,
además de quedarnos con la columna de diabetes:"""
# Cargar el archivo 'CCAA_Enfermedades'
archivo_excel = 'CCAA_Enfermedades.xlsx'
df_enfermedades = pd.read_excel(archivo_excel)

# Encontrar la fila que contiene 'CIFRAS RELATIVAS'
indice_cifras_relativas = df_enfermedades[df_enfermedades.iloc[:, 0] == 'CIFRAS RELATIVAS'].index[0]

# Crear un nuevo DataFrame con las dos primeras filas de encabezados y todo lo que hay debajo de 'CIFRAS RELATIVAS'
nuevo_df_enfermedades = pd.DataFrame(df_enfermedades.iloc[:1, :].values.tolist() + df_enfermedades.iloc[indice_cifras_relativas + 1:, :].values.tolist(), columns=df_enfermedades.columns)

# Seleccionar solo las columnas que contienen al menos una ocurrencia de 'No'
nuevo_df_enfermedades = nuevo_df_enfermedades.loc[:, nuevo_df_enfermedades.isin(['No']).any()]
# Eliminar las dos primeras filas
nuevo_df_enfermedades = nuevo_df_enfermedades.iloc[2:]


# Lista de nuevos encabezados
new_headers_enfermedades = ['Hipertensión arterial', 'Colesterol elevado', 'Diabetes',
                            'Asma, bronquitis crónica o enfisema', 'Enfermedad del corazón', 'Úlcera de estómago',
                            'Alergia', 'Depresión', 'Otras enfermedades mentales',
                            'Jaquecas, migrañas, dolores de cabeza', 'Mala circulación',
                            'Hernias', 'Artrosis y problemas reumáticos', 'Osteoporosis',
                            'Problemas del período menopáusico (excepto osteoporosis)', 'Problemas de la próstata']

# Cambiar el encabezado de cada columna
nuevo_df_enfermedades.columns = new_headers_enfermedades

# Calcular la media de cada fila y agregar la columna 'Physical Health Conditions'
df_ccaa['Physical Health Conditions'] = nuevo_df_enfermedades.mean(axis=1).reset_index(drop=True)

# Copiar la columna 'Diabetes'
df_ccaa['Diabetes'] = nuevo_df_enfermedades['Diabetes'].reset_index(drop=True)

"""4. Modificamos el Excel de alimentación para elaborar un índice de la alimentación de las personas."""

# Cargar el archivo 'CCAA_Alimentacion.xlsx'
file_alimentacion = 'CCAA_Alimentacion.xlsx'

# Leer el archivo excluyendo la tercera fila
df_alimentacion = pd.read_excel(file_alimentacion, header=[0, 1], skiprows=[2])

# Obtener nombres de comunidades autónomas
ccaa_column = df_alimentacion.iloc[:, 0]

# Crear un nuevo DataFrame para realizar la transformación
df_transformed = pd.DataFrame({'CCAA': ccaa_column})

# Iterar sobre las columnas de alimentos y realizar la transformación
for col in df_alimentacion.columns.levels[0][1:]:
    values = df_alimentacion[col]

    # Realizar la transformación
    transformed_values = values['A diario'] * 3 + values['3 o más veces a la semana pero no a diario'] * 2 + \
                         values['1 o 2 veces a la semana'] + values['Menos de 1 vez a la semana'] * -1 + \
                         values['Nunca'] * -2

    # Agregar al nuevo DataFrame
    df_transformed[col] = transformed_values

# Definir las columnas que contienen 'Total'
total_columns = ['Total']

# Dividir cada valor de todas las columnas entre el primer valor de esa columna
df_transformed.iloc[:, 1:] = df_transformed.iloc[:, 1:].div(df_transformed.iloc[0, 1:])

# Eliminar espacios al inicio de los valores de la columna 'CCAA'
df_transformed['CCAA'] = df_transformed['CCAA'].str.replace(r'^\s+', '', regex=True)
# Eliminar la fila que tiene 'Total' como valor de la columna 'CCAA'
df_transformed = df_transformed[df_transformed['CCAA'] != 'Total']

# Lista de nuevos encabezados
new_headers_food = ['CCAA', 'Aperitivos', 'Carne', 'FastFood', 'Dulces', 'Embutidos', 'Fruta', 'Huevos', 'Legumbres', 'Pan',
               'Pasta', 'Pescado', 'Lacteos', 'Refrescos', 'Verduras', 'Zumos']

# Cambiar el encabezado de cada columna
df_transformed.columns = new_headers_food


# Crear un nuevo DataFrame
df_final = pd.DataFrame({'CCAA': df_transformed['CCAA']})

# Definir la fórmula para la columna 'Healthy Eating'
formula1 = (df_transformed['Carne'] +
           (df_transformed['FastFood'] * (-3)) +
           (df_transformed['Dulces'] * (-2)) +
           df_transformed['Fruta'] * 2 +
           df_transformed['Huevos'] +
           df_transformed['Pescado'] +
           df_transformed['Verduras'] * 3 +
           df_transformed['Legumbres'] +
           (df_transformed['Refrescos'] * (-1)) +
           df_transformed['Zumos'] +
           df_transformed['Lacteos']) * 100

# Agregar la columna 'Healthy Eating' al nuevo DataFrame
df_final['Healthy Eating'] = formula1

df_ccaa['Healthy Eating'] = df_final['Healthy Eating'].reset_index(drop=True)


"""5. Modificamos el Excel de obesidad y sobrepeso para quedarnos con los datos de 2022."""

# Cargar el archivo 'CCAA_Sobrepeso.xlsx'
file_sobrepeso = 'CCAA_Sobrepeso.xlsx'

# Leer el archivo excluyendo la segunda fila de encabezados
df_Sobrepeso = pd.read_excel(file_sobrepeso, header=[0, 1], skiprows=[1])

# Eliminar los números de la primera columna y conservar solo los caracteres
df_Sobrepeso.iloc[:, 0] = df_Sobrepeso.iloc[:, 0].str.replace(r'\d+', '')

# Agrupar por la columna 'CCAA' y sumar los valores correspondientes a 2022
df_Sobrepeso_final = df_Sobrepeso.groupby(df_Sobrepeso.columns[0])['2022'].sum().reset_index()

# Renombrar las columnas
df_Sobrepeso_final.columns = ['CCAA', 'Obesidad y sobrepeso']

df_Sobrepeso_final = df_Sobrepeso_final.iloc[:-2, :]
df_ccaa['Obesidad y sobrepeso'] = df_Sobrepeso_final['Obesidad y sobrepeso'].reset_index(drop=True)

"""6. Modificamos el Excel de actividad física para hacer un índice de actividad física por Comunidad Autonoma."""

# Cargar el archivo 'CCAA_ActividadFisica.xlsx'
file_ActFisica = 'CCAA_ActividadFisica.xlsx'

# Leer el archivo excluyendo la segunda fila de encabezados
df_ActividadFisica = pd.read_excel(file_ActFisica, header=[0], skiprows=[1])


# Realizar la operación para crear la nueva columna
df_ActividadFisica['Physical Activity'] = df_ActividadFisica['Nivel alto'] * 1.5 + df_ActividadFisica['Nivel moderado']

# Crear un nuevo DataFrame con las columnas 'CCAA' y 'Physical Activity'
df_fisica = pd.DataFrame({'CCAA': df_ActividadFisica.iloc[:, 0], 'Physical Activity': df_ActividadFisica['Physical Activity']})

# Eliminar la primera fila del nuevo DataFrame
df_fisica = df_fisica.iloc[1:]

df_ccaa['Physical Activity'] = df_fisica['Physical Activity'].reset_index(drop=True)

"""7. Modificamos el Excel de empleados para sacar el porcentaje de trabajadores de oficinas por CCAA"""


# Cargar el archivo 'CCAA_Empleos.xlsx'
file_path_business = 'CCAA_Empleos.xlsx'

# Leer el archivo excluyendo la tercera fila de encabezados
df_Empleos = pd.read_excel(file_path_business, header=[0, 1], skiprows=[2])

# Seleccionar la columna '4 Empleados contables, administrativos y otros empleados de oficina' para el año 2022
empleos_oficina_2022 = df_Empleos['4 Empleados contables, administrativos y otros empleados de oficina', '2022']

# Crear un nuevo DataFrame con las columnas 'CCAA' y 'Empleos Oficina'
df_Empleos_final = pd.DataFrame({'CCAA': df_Empleos.iloc[:, 0], 'Empleos Oficina': empleos_oficina_2022})

df_Empleos_final = df_Empleos_final.iloc[1:]


df_ccaa['Percentage of Business Employees'] = df_Empleos_final['Empleos Oficina'].reset_index(drop=True)



""" 8. Modificamos el Excel de Rentas para quedarnos únicamente con la Renta media por persona del año 2022
con alquiler imputado:"""

# Cargar el archivo
rentas_file_path = 'Rentas_CCAA.xlsx'

# Especificar que la primera fila es el índice y que hay múltiples encabezados
df_Rentas = pd.read_excel(rentas_file_path, index_col=0, header=[0, 1])

# Eliminar valores N/A
df_Rentas = df_Rentas.dropna()

# Eliminar los números de la primera columna y conservar solo los caracteres
df_Rentas.index = df_Rentas.index.str.replace(r'\d+', '')

# Normalizar los encabezados antes de eliminar columnas
df_Rentas.columns = df_Rentas.columns.map(lambda x: ' '.join(map(str, x)))

# Filtrar solo las columnas cuyo encabezado principal es 'Renta media por persona'
filtered_columns_rentas = [col for col in df_Rentas.columns if 'Renta media por persona (con alquiler' in col]
df_Rentas = df_Rentas[filtered_columns_rentas]
df_Rentas = df_Rentas.iloc[1:-2]

# Seleccionar únicamente la columna del año 2022
columna_2022 = df_Rentas['Renta media por persona (con alquiler imputado)  2022']

# Crear un nuevo DataFrame solo con la columna del año 2022
df_Rentas_2022 = pd.DataFrame({'CCAA': df_Rentas.index, 'Renta_2022': columna_2022.values})

# Eliminar la primera fila del nuevo DataFrame
df_Rentas_2022['CCAA'] = df_ccaa['CCAA']


# Seleccionar solo columnas numéricas excluyendo 'Wadges'
numeric_columns = df_ccaa.select_dtypes(include='number')

# Gráfico de dispersión de 'Physical Activity'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Percentage of Business Employees', y='Healthy Eating', data=df_ccaa)
plt.title('Dispersion graph between Business and Healthy eating')
plt.xlabel('% of Business employees')
plt.ylabel('Healthy eating')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()

# Calcular los límites superior e inferior para cada variable numérica
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Aplicar el filtro para eliminar outliers
outlier_filter = ((numeric_columns >= lower_limit) & (numeric_columns <= upper_limit)).all(axis=1)
df_no_outliers = df_ccaa[outlier_filter]

# Gráfico de caja y bigotes después de eliminar outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_no_outliers.drop('CCAA', axis=1))
plt.title('Gráficos de Caja y Bigotes sin Outliers', fontsize=16)
plt.xlabel('Variables', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x para mejorar la legibilidad
plt.tight_layout()
plt.show()

# Añade la columna 'Renta_2022' al DataFrame final
df_ccaa['Wadges'] = df_Rentas_2022['Renta_2022']


# df_ccaa.to_excel('BaseFinalCCAA.xlsx', index=False)
# Establecer la primera columna como índice
df_ccaa.set_index(df_ccaa.columns[0], inplace=True)

correlation_matrix = df_ccaa.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de correlación de todas las variables')
plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()