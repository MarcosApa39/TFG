import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Cargar el archivo Excel
df_Regions = pd.read_excel('Health_Index_Score_UK.xlsx')



# Filtrar filas con 'Region' o 'Country' en la columna 'Area Type'
df_Regions = df_Regions[df_Regions['Area Type [Note 3]'].isin(['Region', 'Country'])]

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

# Mostrar el resultado
# Gráficos de barras
plt.figure(figsize=(12, 8))

# Healthy eating
plt.subplot(3, 1, 1)
sns.barplot(x='Region', y='Healthy eating', data=df_Regions)
plt.title('Gráfico de barras de Healthy eating')
plt.xlabel('Region')
plt.ylabel('Healthy eating')

# Diabetes
plt.subplot(3, 1, 2)
sns.barplot(x='Region', y='Diabetes', data=df_Regions)
plt.title('Gráfico de barras de Diabetes')
plt.xlabel('Region')
plt.ylabel('Diabetes')

# Economic condition
plt.subplot(3, 1, 3)
sns.barplot(x='Region', y='Economic condition', data=df_Regions)
plt.title('Gráfico de barras de Economic condition')
plt.xlabel('Region')
plt.ylabel('Economic condition')

plt.tight_layout()  # Ajusta automáticamente el diseño para mejorar la legibilidad
plt.show()
