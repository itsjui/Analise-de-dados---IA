import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carregando o dataset

dados = pd.read_csv('Titanic-Dataset.csv')

print(f"Visualização das colunas do Dataset")
print(f"\n")
print(dados.columns)
print(f"\n")

# Pré-processamento de dados

# Excluindo colunas desnecessárias
dados = dados.drop(columns = ['Name', 'Embarked', 'Sex', 'Ticket', 'Cabin', 'PassengerId'])
# Preenchendo os dados faltantes na coluna 'Age' com a média
dados.fillna({'Age': dados['Age'].mean()}, inplace=True)

print(f"Correlação -> encontrando as caracteristícas que mais impactam")
corr = dados.corr()
print(f"\n")
print(corr)
print(f"\n")
#sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title("Correlação")

# As variáveis que possuem maior correlação são Age e Fare: 0.091566

# Pré-processando os dados e Selecionando as Features

# Selecionando as features desejadas
selected_features = ['Age', 'Fare']

# Escalonando os dados
scaler = StandardScaler()
dados_scaled = scaler.fit_transform(dados[selected_features])

# Criando um DataFrame com os dados escalonados
dados_scaled = pd.DataFrame(dados_scaled, columns=selected_features)

print(f"Verificando o resultado")
print(f"\n")
print(dados_scaled.head())
print(f"\n")


# Aplicando K-MEANS

# Inicializando um objeto K-Means
kmeans = KMeans(n_clusters=5, random_state=42)

# Dando fit no modelo
kmeans.fit(dados_scaled)

# Pegando as labels dos clusters
cluster_labels = kmeans.labels_

# Adicionando as labels nos dados originais
dados_scaled['Cluster'] = cluster_labels

print(f"Mostrando as primeiras linhas do novo DataFrame com as labels dos clusters")
print(f"\n")
print(dados_scaled.head())
print(f"\n")

# Scatter plot da 'Age' vs 'Fare', colorido por clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Cluster', data=dados_scaled, palette='viridis')
plt.title('K-means Clustering of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

print(f"\n")
print(f"Soma dos quadrados das distâncias")
print(f"\n")
kmeans.inertia_