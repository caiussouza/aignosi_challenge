import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from io import BytesIO

# Leitura do dataset a partir do zip
zip_path = "/home/caiussouza/Documentos/AI/aignosi_challenge/data/mining_data.zip"
csv_path = "MiningProcess_Flotation_Plant_Database.csv"
with zipfile.ZipFile(zip_path, "r") as zip_file:
    with zip_file.open(csv_path) as csv_file:
        dataset = BytesIO(csv_file.read())
        raw_df = pd.read_csv(dataset)

df = raw_df.copy()
# Dimensão dos dados -> 737.453 entradas e 24 atributos
print(df.shape)

# Significado de cada coluna
print(df.columns)
"""

1º Datas
2º e 3º Qualidade do minério antes do processamento
4º a 8º Variáveis importantes para a qualidade do processo
9º a 22º Outros fatores consideráveis para o processo
23 e 24º Qualidade final do minério, sendo o % Silica Concentrate um target.

Algumas observações são tomadas a cada 20 segundos. Outras são por hora.
Qual o tipo dos dados?
Os datos estão corretos? Como validá-los? Ocorreu erros em sensores? (Ponto positivo - prevenção de falhas nos sensores)
Quais são por segundo e quais são por hora?
Quais variáveis têm mais impacto no % silica concentrate?
É possível reduzir a dimensionalidade (PCA e correlação)? Há features muito semelhantes entre si?


src: Kaggle
"""


# Analisando as 5 primeiras linhas
df.head()
"""
Os decimais estão separados por vírgula. Não constam os minutos
e segundos da medição, apenas as horas.

"""

# Analisando os tipos dos dados
df.info()
"""
Todas as colunas são do tipo pandas object.

"""
# Analisando dados nulos
print(df.isnull().sum())
"""
Não há valores nulos.
"""

# Conversão dos tipos de dados para análises estatísticas
for column in df.columns:
    if column != "date":
        df[column] = df[column].str.replace(",", ".").astype(float)
    elif column == "date":
        df[column] = pd.to_datetime(df[column])
df.set_index("date", inplace=True)
# Análise estatística das variáveis
df.describe()
# Nessa parte, vale a pena visualizar os dados e remover possíveis outliers

# Análise da qualidade pré-processamento (2 primeiras colunas)
plt.plot(df.iloc[:, 0:2])

mat_corr = df.corr()
sns.heatmap(mat_corr)
# -0.97 de correalação. Em um modelo, droparia uma delas pra reduzir multicolinearidade
# ou deixaria de usar esse sensor. em última análise, seria uma redução de custos. óbvio q continuaria
# sendo monitorado, porém poderia ter uma redundância benéfica, ou seja, no caso de um sensor falhar, poder usar
# o outro, nem q seja nesse contexto de previsão de qualidade

# É possivel visualizar períodos de não atualização dos sensores, sendo maior entre o meio de maio até o meio de junho

# Analisando sinal target

target = df.iloc[:, -1]
target.head()
target.info()
target.describe()

# Que distribuição é essa?
sns.displot(target)

# Tem outliers?
sns.boxplot(target)


# Quantos outliers? % de outliers
def count_outliers(feature, z_score_threshold=3):
    """
    Counts outliers based on z_score metrics and threshold.
    """
    mean = np.mean(feature)
    std = np.std(feature)
    counter = 0
    for sample in feature:
        if ((sample - mean) / std) >= z_score_threshold:
            counter += 1
    return counter


for column in df.columns:
    print(column)
    print(count_outliers(df[column], z_score_threshold=3))
for column in df.columns:
    print(column)
    print((count_outliers(df[column], z_score_threshold=3) / df[column].size) * 100)

sns.displot(df.iloc[:, 0:2])

sns.lineplot(target)

target.plot(figsize=(32, 16))
