# TFG

# DESCRIPTION:
This repository contains the codes used in the ETL process and the development of the models of the TFG. The files used in the ETL process are:
- BaseNaked.py: here it is developed the ETL process of the database provided by Naked&Sated. The result of this process will be used in the main.py in order to merge both Naked&Sated final database and the Spanish regions database to create a new dataframe that will be used in the file AnalisisDelDato.py.
- Comunidades_Autonomas.py: in this file takes place the ETL process of the Spanish regions databases that contain data related to healthy people domain, life satisfaction, healthy eating, physical activity and diseases of the Spanish population. The result is a new database that will be used in the file main.py in order to merge it with Naked&Sated final database, and in the AnalisisDelDato.py.
- UK_job.py: in this file takes place the ETL process of the England regions databases that contain data related to healthy people domain, life satisfaction, healthy eating, physical activity, economic situation and diseases of the English population. The result is a new database that will be used in the AnalisisDelDato.py file, where the models are developed.
- main.py: in this file both final Naked&Sated database and Spanish Regions database are merged to create a new database.
- AnalisisDelDato.py: in this file takes place the development of the main part of the project: the model application. Principal Components Analysis (PCA), K-Means, KNN, and Gradient Boosting are the models implemented in this project, applied to the databases mentioned before.


# PACKAGES:
In order to be able to use correctly this module you should import these packages or modules:
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Scipy
- Sklearn
- Geopandas
  
You must import this packages with the following commands:
- import pandas as pd
- from sklearn.decomposition import PCA
- from sklearn.preprocessing import StandardScaler
- import matplotlib.pyplot as plt
- import numpy as np
- from sklearn.cluster import KMeans
- from sklearn.metrics import silhouette_score, calinski_harabasz_score
- import seaborn as sns
- from sklearn.preprocessing import MinMaxScaler
- from sklearn.ensemble import GradientBoostingClassifier
- from sklearn.preprocessing import LabelEncoder
- from sklearn.model_selection import GridSearchCV, StratifiedKFold
- from sklearn.neighbors import KNeighborsClassifier
- import geopandas as gpd
- from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
- import seaborn as sns

# LICENSE:
Copyright (c) 2024 Marcos Aparicio Blázquez. Consult 'LICENSE' for more details.
