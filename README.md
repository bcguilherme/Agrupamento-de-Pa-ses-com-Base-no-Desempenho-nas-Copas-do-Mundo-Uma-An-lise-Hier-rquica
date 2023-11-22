# Agrupamento-de-Pa-ses-com-Base-no-Desempenho-nas-Copas-do-Mundo-Uma-An-lise-Hier-rquica

Contextualização

Objetivo Geral:

Este projeto tem como objetivo agrupar países com base em seus desempenhos nas Copas do Mundo de 1930 a 2018, utilizando a técnica de clusterização hierárquica.

Dataset:

Os dados foram coletados a partir deste link e consistem em vários arquivos CSV, um para cada edição da Copa do Mundo, além de um arquivo de resumo que não será utilizado neste caso.

Dicionário dos Dados:
Variável	Descrição
Position	Posição (classificação na copa)
Team	Time
Games Played	Número de jogos disputados pelo time na copa
Win	Número de jogos vencidos pelo time na copa
Draw	Número de jogos empatados pelo time na copa
Loss	Número de jogos perdidos pelo time na copa
Goals For	Número de gols marcados pelo time na copa
Goals Against	Número de gols sofridos pelo time na copa
Goals Difference	Diferença entre gols marcados e gols sofridos pelo time na copa
Points	Pontuação do time na copa
Instruções

Pré-processamento dos dados:
Realizar pré-processamento para tornar os dados comparáveis entre os times.
Plotagem de Dendrograma:
Utilizar diferentes métodos da abordagem bottom-up para criar dendrogramas.
Inspecionar os gráficos e tirar conclusões.
Avaliação da Quantidade "Ideal" de Clusters:
Avaliar a quantidade "ideal" de clusters utilizando métricas estudadas no módulo.
Criação de Grupos e Descrição:
Escolher uma quantidade de grupos e descrever os grupos estudados, incluindo os times presentes em cada grupo.
Considerar apenas os times que participaram de mais de 3 Copas.
Setup

Bibliotecas

python
Copy code
import re
import glob
import numpy as np
import pandas as pd
from ipywidgets import interact
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import MinMaxScaler, minmax_scale, scale
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt
Funções Personalizadas

python
Copy code
def read_csv_custom(file):
    df = pd.read_csv(file)
    df['Year'] = int(re.search('\d+', file).group())
    return df
Carregamento dos Dados

python
Copy code
# Download e descompactação dos arquivos
!wget https://github.com/cmcouto-silva/datasets/raw/main/datasets/fifa-football-world-cup-dataset.zip
!unzip fifa-football-world-cup-dataset.zip

# Listagem dos arquivos
files = glob.glob(r'FIFA - [1-2]*')

# Concatenação dos dados de diferentes Copas
df_teams = pd.concat([read_csv_custom(file) for file in files])
df_teams = df_teams.sort_values(['Year', 'Position'])
Pré-processamento

python
Copy code
# Adição de novas colunas para comparabilidade
df_teams = (
  df_teams
  .assign(**{
     'Win %': lambda x: x['Win'] / x['Games Played'],
     'Draw %': lambda x: x['Draw'] / x['Games Played'],
     'Loss %': lambda x: x['Loss'] / x['Games Played'],
     'Avg Goals For': lambda x: x['Goals For'] / x['Games Played'],
     'Avg Goals Against': lambda x: x['Goals For'] / x['Games Played'],
     'Normalized Rank': lambda x: x.Position.rank(pct=True, ascending=False),
     'Normalized Points': lambda x: x.Points.rank(pct=True, ascending=False),
     'Century': lambda x: (x.Year // 100 + 1).astype(str)
  })
)

# Agrupamento por time (considerando apenas aqueles com mais de 3 Copas)
df_teams_stats = df_teams.groupby('Team').agg(
  n_cups = ('Team', 'count'),
  n_games = ('Games Played', 'sum'),
  avg_wins_per_game = ('Win %', 'mean'),
  avg_draws_per_game = ('Draw %', 'mean'),
  avg_losses_per_game = ('Loss %', 'mean'),
  avg_goals_for = ('Avg Goals For', 'mean'),
  avg_rank = ('Normalized Rank', 'mean')
)

df_teams_stats = df_teams_stats.query('n_cups > 3')
Visualização de Outliers

python
Copy code
# Verificação de outliers
df_teams_stats.apply(scale).boxplot()
plt.xticks(rotation=90, ha='right');
Limitação de Outliers

python
Copy code
# Limitação de outliers para no máximo 3 desvios padrões
for col in df_teams_stats.columns:
  avg, std = df_teams_stats[col].agg(['mean', 'std'])
  df_teams_stats[col] = df_teams_stats[col].clip(lower=avg-3*std, upper=avg+3*std)
Normalização para uma Mesma Escala

python
Copy code
# Normalização para uma mesma escala (-1 a 1)
scaler = MinMaxScaler(feature_range=(-1,1))
X = df_teams_stats.copy()
X[:] = scaler.fit_transform(X)
Modelagem e Visualização da Clusterização Hierárquica

python
Copy code
# Modelagem
plt.figure(figsize=(21,12))
Z = linkage(X, method='ward')
dendrogram_dict = dendrogram(Z, labels=X.index)
plt.xticks(fontsize=14)
plt.show()
python
Copy code
# Teste de diferentes métodos de agrupamento hierárquico bottom-up
@interact(method=['ward','complete','single','average','weighted','centroid','median'])
def hplot(method):
  plt.figure(figsize=(20,10))
  Z = linkage(X, method=method)
  dendrogram_dict = dendrogram(Z, labels=X.index)
  sns.despine(bottom=True, trim=True)
  plt.xticks(fontsize=14)
  plt.show()
python
Copy code
# Teste de diferentes valores de corte para o método "ward"
@interact(color_threshold=[5,4,2.8,2.5])
def hplot(color_threshold):
  plt.figure(figsize=(20,10))
  Z = linkage(X, method='ward')
