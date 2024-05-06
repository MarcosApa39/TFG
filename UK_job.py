import pandas as pd

file = 'UK_Jobs.xlsx'

# Importamos el archivo Excel
df_Jobs = pd.read_excel(file, header=[0, 1, 2])

# Rellenar valores N/A con el valor de la celda superior
df_Jobs = df_Jobs.ffill()

# Eliminar valores N/A
df_Jobs = df_Jobs.dropna()

# Normalizar los encabezados antes de eliminar columnas
df_Jobs.columns = df_Jobs.columns.map(lambda x: ' '.join(map(str, x)))

# Filtrar solo las columnas cuyo encabezado principal es 'Total', 'Region' o 'Industry'
filtered_columns = [col for col in df_Jobs.columns if ('Total' in col or 'Industry' in col or 'Region' in col)]

df_Jobs = df_Jobs[filtered_columns]
# Lista de nuevos encabezados
new_headers = ['Region', 'Industry', 'Count', 'Employment', 'Employees']

# Cambiar el encabezado de cada columna
df_Jobs.columns = new_headers





# Crear un DataFrame df_final_jobs con la columna 'Region'
df_final_jobs = pd.DataFrame({'Region': df_Jobs['Region'].unique()})

# Filtrar df_Jobs para obtener solo las filas con 'Industry' igual a 'Professional and Business'
df_filtered = df_Jobs[df_Jobs['Industry'] == 'Professional and Business']

# Crear la columna 'Business Employees' en df_final_jobs
df_final_jobs['Business Employees'] = df_filtered['Employees'].values

# Calcular la suma de 'Employees' por 'Region' en df_Jobs
region_employees_sum = df_Jobs.groupby('Region')['Employees'].sum().reset_index()

# Merge entre df_final_jobs y region_employees_sum para obtener la columna 'Percentage of Business Employees'
df_final_jobs = pd.merge(df_final_jobs, region_employees_sum, on='Region', how='left')
df_final_jobs['Percentage of Business Employees'] = (df_final_jobs['Business Employees'] / df_final_jobs['Employees']) * 100

# Eliminar la columna 'Employees' de df_final_jobs si no se necesita
df_final_jobs = df_final_jobs.drop('Employees', axis=1)

# Mostrar el DataFrame final
# print(df_final_jobs)








# Cargar el archivo Excel
df_Regions = pd.read_excel('Health_Index_Score_UK.xlsx')



# Filtrar filas con 'Region' o 'Country' en la columna 'Area Type'
df_Regions = df_Regions[df_Regions['Area Type [Note 3]'].isin(['Region', 'Country'])]

# Seleccionar las columnas específicas
selected_columns = ['Area Name', 'Healthy People Domain', 'Life satisfaction [Pe4]',
                    'Physical health conditions [Pe]', 'Diabetes [Pe5]', 'Healthy eating [L1]',
                    'Overweight and obesity in adults [L3]',
                    'Physical activity [L1]']

df_Regions = df_Regions[selected_columns]

new_headers_df = ['Region', 'Healthy People Domain', 'Life satisfaction',
               'Physical health conditions', 'Diabetes', 'Healthy eating',
               'Overweight and obesity in adults',
               'Physical activity']

df_Regions.columns = new_headers_df

























# Merge de los DataFrames utilizando la columna 'Region' como clave primaria
df_merged = pd.merge(df_Regions, df_final_jobs, left_on='Region', right_on='Region', how='inner')

# Mostrar el DataFrame resultante
df_merged.to_excel("UK_Final_DB.xlsx", index=False)


