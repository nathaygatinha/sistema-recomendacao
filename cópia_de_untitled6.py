# -*- coding: utf-8 -*-
"""Cópia de Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ceJQvXMCvj2O2M8RWcT43tDAk2hogs38
"""



# Importando as bibliotecas necessárias
import numpy as np
from sklearn.cluster import KMeans

# Matriz com os filmes assistidos
filmes_assistidos = np.array([
    [1,0,0,1],
    [1,1,0,0],
    [0,1,1,0],
    [0,0,1,1],
    [1,0,1,0],
    [0,1,0,1]
])

# Definindo o número de clusters (Grupos)
num_cluster = 2

# Inicializar o modelo
kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=10)

# Treinando o modelo
kmeans.fit(filmes_assistidos)

# Classificando os usuários
grupos_indice = kmeans.predict(filmes_assistidos)

# Exibir os dados
print("Usuário pertence ao seguinte grupo:")
for i, cluster in enumerate(grupos_indice):
      print(f"Usuário {i+1} pertence ao grupo {cluster+1}")

print("\nFilmes assistidos:")
for i in range(len(filmes_assistidos)):
    assistidos = np.where(filmes_assistidos[i] == 1)[0] + 1
    print(f"usuário(i+1 assistiu aos filmes:{assistidos})")

return sorted(filmes_r)

def multiplicacao(num1, num2) :
  mult = num1 * num2
  return mult

  print (multiplicacao(7,10))
  print (multiplicacao(8,10))