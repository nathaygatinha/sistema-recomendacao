# Importando bibliotecas necessárias
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

#1º Passo: especificar o caminho do arquivo CSV
caminho_arquivo = '/content/filmes_100_usuarios.csv'

# 2º Passo: Ler o CVS usando pandas
df = pd.read_csv(caminho_arquivo)

# Exibir o cabeçalho do arquivo para verificar se foi lido corretamente
print(df.head())

# Matriz com os filmes assistidos
filmes_assistidos = df.drop(columns=["Unnamed: 0"]).values

# Definindo o número de clusters (Grupos)
num_cluster = 2

# Inicializar o modelo
kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=10)

# Treinado o modelo
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
   print(f"Usuário {i+1} assistiu aos filmes: {assistidos}")

# Função que recomeda filmes
def recomendar_filmes(filmes, filmes_assistidos, grupos_indice):

  filmes = np.array(filmes)
  # Encontrar o grupo do usuário com base em seu vetor de filmes assistidos
  usuário_id = len(filmes_assistidos)
  grupo_usuário = kmeans.predict([filmes])[0]

  # Encontrar todos os usuários do mesmo grupo
  usuário_no_mesmo_grupo = [i for i in range(len(grupos_indice)) if grupos_indice[i] == grupo_usuário]

  # Filmes assitidos pelos usuários no mesmo grupo
  filmes_recomendados = set()
  for usuário in usuário_no_mesmo_grupo:
    filmes_assistidos_usuário = np.where(filmes_assistidos[usuário] == 1)[0]
    filmes_recomendados.update(filmes_assistidos_usuário)

  # Remover filmes que o usuário já assitiu
    filmes_recomendados = filmes_recomendados - set(np.where(filmes == 1)[0])

  # Ajustar os índices dos filmes recomendados (de volta para 1-based)
  filmes_recomendados = [filme + 1 for filme in filmes_recomendados]

  return sorted(filmes_recomendados)

# Exemplo de uso de função recomendar filmes
filmes_assistidos_usuario = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0] # Vetor de filmes
#assistidos (por exemplo, assistiu aos filmes 1 e 3)
filmes_recomendados = recomendar_filmes(filmes_assistidos_usuario, filmes_assistidos, grupos_indice)

print(f"\nFilmes recomendados: {filmes_recomendados}")