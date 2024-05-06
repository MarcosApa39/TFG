# LIBRERÍAS UTILIZADAS:
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import geopandas as gpd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# FUNCIONES CREADAS:
def perform_pca_extended(file_path, total_variables_limit=4, variance_threshold=0.9):
    """Esta función aplica el PCA a un dataframe creado a partir del archivo introducido."""
    # Cargar y preparar los datos
    df = pd.read_excel(file_path).drop(columns=['CCAA', 'Importe'])
    data_numeric = df.select_dtypes(include=['float64', 'int64'])

    # Estandarizamos los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    # Aplicamos el PCA sin limitar el número de componentes para que el análisis seacompleto
    pca_full = PCA()
    pca_full.fit(data_scaled)

    # Varianza acumulada explicativa
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Gráfico de la varianza acumulada
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title('Análisis de Varianza Acumulada Explicativa')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Acumulada Explicativa')
    plt.grid(True)
    plt.show()

    # Test de Kaiser-Guttman
    eigenvalues = pca_full.explained_variance_
    n_components_kaiser = sum(eigenvalues > 1)

    # Imprime los resultados del test de Kaiser-Guttman
    print(f"Número de componentes sugeridos por el test de Kaiser-Guttman: {n_components_kaiser}")

    # Aplicamos el PCA con el umbral de varianza para la selección de componentes
    pca = PCA(n_components=variance_threshold)
    pca.fit(data_scaled)

    # Información del PCA
    print(f"Número de componentes principales seleccionados según el umbral de varianza: {pca.n_components_}")
    print(f"Varianza explicada por cada componente principal seleccionado: {pca.explained_variance_ratio_}")

    # Obtener y mostrar loadings
    loadings = pca.components_
    loadings_df = pd.DataFrame(loadings, columns=data_numeric.columns).T
    print("Loadings de cada variable en cada componente principal seleccionado:")
    print(loadings_df)

    # Seleccionamos las variables más importantes basándose en los loadings más altos
    top_variables = loadings_df.abs().max(axis=1).nlargest(total_variables_limit).index.tolist()

    print(f"Variables seleccionadas para el análisis (limitadas a un total de {total_variables_limit}):")
    print(top_variables)

    # Devuelve el número de componentes sugeridos por Kaiser, la varianza acumulada y las variables principales
    return n_components_kaiser, explained_variance, top_variables


def perform_elbow_method(X_scaled, max_clusters=10):
    """Esta función aplica el método Elbow"""

    # Para asegurarse de que el número máximo de clusters no exceda el número de muestras
    n_samples = X_scaled.shape[0]
    max_clusters = min(n_samples, max_clusters)

    # para calcular la suma de las distancias al cuadrado para diferentes números de clusters
    inertia = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Hacemos el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances (Inertia)')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.show()


def perform_silhouette_analysis(X_scaled, range_clusters=8):
    """Esta función sirve para obtener el coeficiente Silhouette"""
    # Cálculo y gráfica del coeficiente
    silhouette_scores = []
    for i in range(2, range_clusters):  # Ajuste del rango para coincidir con el cálculo de silhouette_scores
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, range_clusters), silhouette_scores, marker='o')
    plt.title('Silhouette Score For Different k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, range_clusters))
    plt.grid(True)
    plt.show()


def perform_calinski_harabasz_analysis(X_scaled, range_clusters=8):
    """Esta función sirve para calcular y mostrar el índice Calinsi"""
    calinski_harabasz_scores = []
    for i in range(2, range_clusters):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, range_clusters), calinski_harabasz_scores,
             marker='o')
    plt.title('Calinski Score For Different k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski Score')
    plt.xticks(range(2, 8))
    plt.grid(True)
    plt.show()


def plot_map_with_labels(gdf, title, color_map: dict, column: str):
    """ Función para crear un mapa de colores basado en los valores de 'Predicted Sales Success'"""
    colors = [color_map[value] for value in gdf[column]]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    gdf.plot(color=colors, legend=True, ax=ax)

    # Añadir etiquetas de texto para cada región
    for idx, row in gdf.iterrows():
        plt.text(row.geometry.centroid.x, row.geometry.centroid.y, s=row['Region'],
                 horizontalalignment='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Crear leyenda manualmente
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=val, markersize=10) for key, val in color_map.items()]
    ax.legend(handles=legend_elements, title="Predicted Sales Success")

    ax.set_title(title)
    plt.axis('off')  # Ocultar los ejes
    plt.show()


if __name__ == '__main__':
    # 1. ANÁLISIS DE COMPONENTES PRINCIPALES:
    print("1. ANALISIS DE COMPONENTES PRINCIPALES: \n")
    input("Pulse Enter para continuar...")

    # Ejecutamos la función con la ruta al archivo Excel
    n_components_kaiser, explained_variance, global_top_variables = perform_pca_extended('Naked_CCAA_Merged.xlsx')

    # Imprimimos las variables principales
    print("\n\n\n\n")
    print("Variables principales almacenadas:")
    print(global_top_variables)

    # Pausa para que el usuario pueda ver los resultados detenidamente antes de continuar
    input("Pulse Enter para continuar...")

    # Resultados del análisis
    print(f"Número de componentes sugeridos por Kaiser: {n_components_kaiser}")
    print(f"Varianza acumulada explicativa final: {explained_variance[-1]}")

    # Pausa para que el usuario pueda ver los resultados detenidamente antes de continuar
    input("Pulse Enter para continuar...")

    # 2. CLUSTERIZACIÓN CON K-MEANS:
    print("2. CLUSTERIZACION CON K-MEANS: \n")
    input("Pulse Enter para continuar...")


    # Cargamos las BBDD:
    uk_global_df = pd.read_excel('UK_Global.xlsx')
    uk_regions = pd.read_excel('UK_Final_DB.xlsx')

    # Seleccionamos solo las columnas de interés para el K-Means:
    columns_of_interest = ['Physical health conditions', 'Healthy eating', 'Healthy People Domain', 'Life satisfaction']
    X_global = uk_global_df[columns_of_interest]
    X_regions = uk_regions[columns_of_interest]

    # Normalizamos las variables seleccionadas
    scaler = StandardScaler()
    X_scaled_regions = scaler.fit_transform(X_regions)
    X_scaled_global = scaler.fit_transform(X_global)

    # Utilizamos el método Elbow para saber el número de clústers:
    perform_elbow_method(X_scaled_regions)
    perform_elbow_method(X_scaled_global)

    # Utilizamos, para el dataframe de los distritos, el coeficiente Silhouette y el índice Calinski:
    perform_silhouette_analysis(X_scaled_global)
    perform_calinski_harabasz_analysis(X_scaled_global)

    # Aplicamos el K-means para clasificar las regiones en 3 grupos
    kmeans_regions = KMeans(n_clusters=3, random_state=42)
    kmeans_regions.fit(X_scaled_regions)

    kmeans_global = KMeans(n_clusters=2, random_state=42)
    kmeans_global.fit(X_scaled_global)

    # Centroides de clusters de las regiones
    cluster_centroids_regions = kmeans_regions.cluster_centers_
    cluster_centroids_df_regions = pd.DataFrame(scaler.inverse_transform(cluster_centroids_regions),
                                                columns=columns_of_interest)
    print("Centroides de los clusters:")
    print(cluster_centroids_df_regions)

    # Pausa para que el usuario pueda ver los resultados detenidamente antes de continuar
    input("Pulse Enter para continuar...")

    # Centroides de clusters de los distritos
    cluster_centroids_global = kmeans_global.cluster_centers_
    cluster_centroids_df_global = pd.DataFrame(scaler.inverse_transform(cluster_centroids_global),
                                               columns=columns_of_interest)
    print("Centroides de los clusters:")
    print(cluster_centroids_df_global)

    # Asignamos las etiquetas de los clusters a las regiones
    uk_regions['Cluster'] = kmeans_regions.labels_
    uk_global_df['Cluster'] = kmeans_global.labels_

    # Visualizamos los resultados de la clasificación en un gráfico de barras
    plt.figure(figsize=(14, 7))
    plt.bar(uk_regions['Region'], uk_regions['Cluster'], color='skyblue')
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Cluster Group', fontsize=14)
    plt.title('Cluster Assignment by Region', fontsize=16)
    plt.xticks(rotation=90)
    plt.show()

    # Pausa para que el usuario pueda ver los resultados detenidamente antes de continuar
    input("Pulse Enter para continuar...")

    # Añadimos las etiquetas de cluster
    X_vis_global = uk_global_df[columns_of_interest]
    X_vis_global['Cluster'] = uk_global_df['Cluster']

    X_vis_regions = uk_regions[columns_of_interest]
    X_vis_regions['Cluster'] = uk_regions['Cluster']

    # Generamos el pair plot coloreado por etiquetas de cluster
    sns.pairplot(X_vis_regions, hue='Cluster', vars=columns_of_interest, palette='viridis', diag_kind='kde')
    plt.suptitle('Pair Plot of Regions by Cluster', size=16)
    plt.show()

    # Generamos el pair plot coloreado por etiquetas de cluster
    sns.pairplot(X_vis_global, hue='Cluster', vars=columns_of_interest, palette='viridis', diag_kind='kde')
    plt.suptitle('Pair Plot of UK_Global Regions by Cluster', size=16)
    plt.show()

    # 3. CLASIFICACIÓN CON GRADIENT BOOSTING:
    print("3. CLASIFICACION CON GRADIENT BOOSTING: \n")
    input("Pulse Enter para continuar...")

    # Cargamos los conjuntos de datos
    spain_df = pd.read_excel('Naked_CCAA_Merged.xlsx')
    uk_df = pd.read_excel('UK_Global.xlsx')
    ccaa_df = pd.read_excel('BaseFinalCCAA.xlsx')

    # Renombramos las columnas en el conjunto de datos del Reino Unido para que coincidan con las de España
    uk_df.rename(columns={
        'Life satisfaction': 'Life Satisfaction',
        'Physical health conditions': 'Physical Health Conditions',
        'Physical activity': 'Physical Activity',
        'Healthy eating': 'Healthy Eating',
        'Economic condition': 'Wadges'
    }, inplace=True)

    # Seleccionamos y normalizamos características relevantes en el conjunto de datos de España
    common_features_spain = ['Life Satisfaction', 'Physical Health Conditions', 'Healthy Eating', 'Physical Activity',
                             'Diabetes', 'Wadges']
    spain_features = spain_df[common_features_spain]
    scaler_spain = MinMaxScaler()
    spain_features_normalized = scaler_spain.fit_transform(spain_features)

    # Ajustamos los umbrales para la categorización de 'Sales Success' basándonos en percentiles 25 y 75
    percentiles_importe = np.percentile(spain_df['Importe'], [10, 90])
    spain_df['Sales Success'] = pd.cut(spain_df['Importe'],
                                       bins=[-np.inf, percentiles_importe[0], percentiles_importe[1], np.inf],
                                       labels=['Low', 'Medium', 'High'])

    # Preparamos las características y la variable objetivo para el modelo
    X_train_spain = spain_features_normalized
    y_train_spain = spain_df['Sales Success']

    # Codificamos la variable objetivo
    label_encoder_spain = LabelEncoder()
    y_train_spain_encoded = label_encoder_spain.fit_transform(y_train_spain)

    # Definimos el modelo y los hiperparámetros a ajustar
    gb_model_spain = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.2]
    }

    # Configurar StratifiedKFold
    min_samples_in_class = min(y_train_spain_encoded, key=list(y_train_spain_encoded).count)
    n_splits = max(2, min_samples_in_class)

    cv = StratifiedKFold(n_splits=n_splits)

    # Búsqueda de cuadrícula con validación cruzada usando StratifiedKFold
    grid_search = GridSearchCV(gb_model_spain, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_spain, y_train_spain_encoded)

    best_model = grid_search.best_estimator_

    # Seleccionar y normalizar características relevantes en el conjunto de datos del Reino Unido
    uk_features = uk_df[common_features_spain]
    uk_features_normalized = scaler_spain.transform(uk_features)

    # Predecir el potencial de éxito en las regiones del Reino Unido con el mejor modelo
    uk_predictions = best_model.predict(uk_features_normalized)

    # Decodificar las predicciones para obtener las etiquetas originales
    uk_predictions_labels = label_encoder_spain.inverse_transform(uk_predictions)

    # Asignar las predicciones a la columna 'Predicted Sales Success' en el DataFrame del Reino Unido
    uk_df['Predicted Sales Success'] = uk_predictions_labels

    # Cargar el archivo GeoJSON de los distritos de Londres
    london_districts = gpd.read_file('london_map.geojson')

    # Fusionar los datos geográficos con los datos del DataFrame `uk_df` en función de los nombres de los distritos
    london_map = london_districts.merge(uk_df, left_on='name', right_on='Region')

    # Definir un diccionario para mapear los valores de 'Predicted Sales Success' a colores
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}

    # Dibujar el mapa de los distritos de Londres con colores y nombres de las regiones
    plot_map_with_labels(london_map, 'Predicted Sales Success in London Districts', color_map,
                         'Predicted Sales Success')

    # Evaluación en el conjunto de entrenamiento (España)
    spain_train_pred = best_model.predict(X_train_spain)

    # Métricas para el conjunto de entrenamiento
    accuracy_train = accuracy_score(y_train_spain_encoded, spain_train_pred)
    precision_train = precision_score(y_train_spain_encoded, spain_train_pred, average='macro', zero_division=0)
    recall_train = recall_score(y_train_spain_encoded, spain_train_pred, average='macro', zero_division=0)
    f1_train = f1_score(y_train_spain_encoded, spain_train_pred, average='macro', zero_division=0)

    # Visualización de la matriz de confusión para el conjunto de entrenamiento
    conf_matrix_train = confusion_matrix(y_train_spain_encoded, spain_train_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_train, annot=True, fmt='g', cmap='Pastel1')
    plt.title('Matriz de Confusión - Conjunto de Entrenamiento (España)')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Verdaderos')
    plt.show()

    print(f"Conjunto de Entrenamiento (España) - Exactitud: {accuracy_train}, Precisión: {precision_train}, Recuperación: {recall_train}, F1: {f1_train}")

    # Pausa para que el usuario pueda ver los resultados detenidamente antes de continuar
    input("Pulse Enter para continuar...")

    # PARA LAS CC.AA CON GRADIENT BOOSTING:

    # Seleccionamos y normalizar características relevantes en el conjunto de datos de las CCAA
    ccaa_features = ccaa_df[common_features_spain]
    ccaa_features_normalized = scaler_spain.transform(ccaa_features)

    # Predecimos el potencial de éxito en las regiones de las CCAA con Gradient Boosting
    ccaa_predictions = best_model.predict(ccaa_features_normalized)

    # Decodificamos las predicciones para obtener las etiquetas originales
    ccaa_predictions_labels = label_encoder_spain.inverse_transform(ccaa_predictions)

    # Asignamos las predicciones a la columna 'Predicted Sales Success' en el DataFrame del Reino Unido
    ccaa_df['Predicted Sales Success'] = ccaa_predictions_labels


    # 4. CLASIFICACIÓN CON KNN:
    print("4. CLASIFICACION CON KNN: \n")
    input("Pulse Enter para continuar...")

    # Definimos el modelo KNN y los hiperparámetros a ajustar
    knn_model_spain = KNeighborsClassifier()
    param_grid_KNN = {
        'n_neighbors': [3, 5, 7, 9],  # Número de vecinos
        'weights': ['uniform', 'distance'],  # Ponderación de los vecinos
        'metric': ['euclidean', 'manhattan']  # Métrica de distancia
    }

    # Configuramos StratifiedKFold
    min_samples_in_class_KNN = y_train_spain_encoded.min()
    n_splits_KNN = max(2, min_samples_in_class_KNN)

    cv_KNN = StratifiedKFold(n_splits=n_splits_KNN)

    # Búsqueda de cuadrícula con validación cruzada usando StratifiedKFold
    grid_search_KNN = GridSearchCV(knn_model_spain, param_grid_KNN, cv=cv_KNN, scoring='accuracy')
    grid_search_KNN.fit(X_train_spain, y_train_spain_encoded)

    best_model_KNN = grid_search_KNN.best_estimator_

    # Predecimos el potencial de éxito en las regiones del Reino Unido con KNN
    uk_predictions_KNN = best_model_KNN.predict(uk_features_normalized)

    # Decodificamos las predicciones para obtener las etiquetas originales
    uk_predictions_labels_KNN = label_encoder_spain.inverse_transform(uk_predictions_KNN)

    # Asignamos las predicciones a la columna 'Predicted Sales Success KNN' en el DataFrame del Reino Unido
    uk_df['Predicted Sales Success KNN'] = uk_predictions_labels_KNN

    # Fusionar los datos geográficos con los datos del DataFrame `uk_df` en función de los nombres de los distritos
    london_map_knn = london_districts.merge(uk_df, left_on='name', right_on='Region')

    # Dibujar el mapa de los distritos de Londres con colores y nombres de las regiones
    plot_map_with_labels(london_map_knn, 'Predicted Sales Success in London Districts', color_map,
                         'Predicted Sales Success KNN')

    # PARA LAS CC.AA CON KNN:

    # Predecimos el potencial de éxito en las regiones de las CCAA con knn
    ccaa_predictions_KNN = best_model_KNN.predict(ccaa_features_normalized)

    # Decodificamos las predicciones para obtener las etiquetas originales
    ccaa_predictions_labels_KNN = label_encoder_spain.inverse_transform(ccaa_predictions_KNN)

    # Asignamos las predicciones a la columna 'Predicted Sales Success KNN' en el DataFrame del Reino Unido
    ccaa_df['Predicted Sales Success KNN'] = ccaa_predictions_labels_KNN

    # Evaluación en el conjunto de entrenamiento (España)
    spain_train_pred_KNN = best_model_KNN.predict(X_train_spain)

    # Métricas para el conjunto de entrenamiento
    accuracy_train_KNN = accuracy_score(y_train_spain_encoded, spain_train_pred_KNN)
    precision_train_KNN = precision_score(y_train_spain_encoded, spain_train_pred_KNN, average='macro', zero_division=0)
    recall_train_KNN = recall_score(y_train_spain_encoded, spain_train_pred_KNN, average='macro', zero_division=0)
    f1_train_KNN = f1_score(y_train_spain_encoded, spain_train_pred_KNN, average='macro', zero_division=0)

    # Visualización de la matriz de confusión para el conjunto de entrenamiento
    conf_matrix_train_KNN = confusion_matrix(y_train_spain_encoded, spain_train_pred_KNN)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_train_KNN, annot=True, fmt='g', cmap='Pastel1')
    plt.title('Matriz de Confusión - Conjunto de Entrenamiento (España)')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Verdaderos')
    plt.show()

    print(
        f"Conjunto de Entrenamiento (España) - Exactitud: {accuracy_train_KNN}, Precisión: {precision_train_KNN}, Recuperación: {recall_train_KNN}, F1: {f1_train_KNN}")

    print("Muchas gracias.\n")









