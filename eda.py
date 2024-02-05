import zipfile
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, ccf
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA


# Plota correlação cruzada entre duas variáveis.
def plot_ccf(ts1, ts2, title="Cross correlation between feature and target"):
    backwards = ccf(ts2, ts1, unbiased=False)[::-1]
    forwards = ccf(ts1, ts2, unbiased=False)
    ccf_output = np.r_[backwards[:-1], forwards]
    max_corr_idx = np.argmax(np.abs(ccf_output))
    max_corr_lag = range(-len(ccf_output) // 2, len(ccf_output) // 2)[max_corr_idx]
    max_corr_value = ccf_output[max_corr_idx]
    plt.stem(range(-len(ccf_output) // 2, len(ccf_output) // 2), ccf_output)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("CCF")
    return max_corr_value, max_corr_lag


# Plota múltiplos histogramas
def mult_hist_plots(data, nrows, ncols, figsize=(12, 8), fontsize=36):
    fig, axs = plt.subplots(4, 6, figsize=figsize)
    for i, column in enumerate(data.columns):
        if ncols == 1:
            ax = axs[i]
        else:
            ax = axs[i // ncols, i % ncols]
        sns.histplot(data[column], ax=ax)
        ax.set_title(column, fontsize=fontsize)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frequência")
    plt.tight_layout()


# Plota múltiplos boxplots
def mult_boxplots(data, nrows, ncols, figsize=(12, 8)):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i, column in enumerate(data.columns):
        if ncols == 1:
            ax = axs[i]
        else:
            ax = axs[i // ncols, i % ncols]
        sns.boxplot(
            data[column],
            ax=ax,
            color="grey",
            linewidth=2,
            flierprops=dict(marker="o", markerfacecolor="red", markersize=8),
        )
        ax.set_title(column, fontsize=36)
    plt.tight_layout()


# Plota múltiplos lineplots
def mult_lineplots(data, nrows, ncols, figsize=(12, 8)):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i, column in enumerate(data.columns):
        if ncols == 1:
            ax = axs[i]
        else:
            ax = axs[i // ncols, i % ncols]
        sns.lineplot(data[column], ax=ax)
        ax.set_title(column, fontsize=36)
    plt.tight_layout()


# Teste Augmented Dickey-Fuller
def adf_test(data, interval=0.05, maxlag=10):
    for column in data.columns:
        adf = adfuller(data[column], maxlag=maxlag)
        if adf[1] < interval:
            print(column, " é estacionária!\np-valor: ", adf[1])
        else:
            print(column, " não é estacionária!\np-valor: ", adf[1])


# Leitura do dataset a partir do zip
zip_path = "/home/caiussouza/Documentos/AI/aignosi_challenge/data/mining_data.zip"
csv_path = "MiningProcess_Flotation_Plant_Database.csv"
with zipfile.ZipFile(zip_path, "r") as zip_file:
    with zip_file.open(csv_path) as csv_file:
        dataset = BytesIO(csv_file.read())
        raw_df = pd.read_csv(dataset)

# Cópia dos dados crus para manejo
df = raw_df.copy()

# Visualização dos primeiros dados
df.head()

# Visualização dos tipos dos dados
df.info()

# Verificação de valores nulos
df.isnull().sum()

# Métricas estatísticas das variáveis
df.describe()

# Conversão dos tipos de dados para numérico
for column in df.columns:
    if column != "date":
        df[column] = df[column].str.replace(",", ".").astype(float)
    elif column == "date":
        df[column] = pd.to_datetime(df[column])
df.set_index("date", inplace=True)

# Adição dos segundos de acordo com a menor período de aquisição
# informado no Kaggle: 20 segundos.
df.index = pd.Timestamp("2017-03-10 01:00:00") + pd.to_timedelta(
    range(0, len(df) * 20, 20), unit="s"
)

# Gráfico de linhas para identificação de anormalidades
plt.title("Mining quality feature")
plt.plot(df)

# Remoção do período de anormalidade anterior a Abril
clean_df = df[df.index > "2017-04-01 01:00:00"]

# Distribuição dos dados
mult_hist_plots(clean_df, 4, 6, figsize=(60, 40))

# Boxplots gerais
mult_boxplots(clean_df, 4, 6, figsize=(60, 40))

# Divisão dos grupos p/ detalhar análise

# Minério pré-processamento
first_group = clean_df.iloc[:, 0:2]
# Variáveis importantes de qualidade
second_group = clean_df.iloc[:, 2:7]
# Variáveis ambientais
third_group = clean_df.iloc[:, 7:21]
# Qualidade pós-processamento
fourth_group = clean_df.iloc[:, 21:23]
# Target
target = clean_df.iloc[:, -1]


"""

Primeiro grupo:

* % Iron Feed
* % Silica Feed

"""

# Visualização das séries temporais
mult_lineplots(first_group, 2, 1, figsize=(30, 20))

# Teste de estacionariedade
adf_test(first_group)

# Correlação entre sinais
corr_mat = first_group.corr()
sns.heatmap(corr_mat, annot=True)
sns.pairplot(first_group)

# Correlação c/ target

max_val, lag = plot_ccf(
    first_group.resample("min").mean().iloc[:, 0],
    target.resample("min").mean(),
    "% Silica Feed x Target",
)
print(f"Maior correlação: {max_val} com {lag/60} horas de diferença")
"""

Segundo grupo:

* Starch flow
* Amina flow
* Ore pulp flow
* Ore pulp pH
* Ore pulp density

"""

# Resample para dia para facilitar visualização
sec_grp_h = second_group.resample("H").mean()

# Visualização das séries temporais
mult_lineplots(sec_grp_h, 5, 1, figsize=(60, 40))

# Teste de estacionariedade
adf_test(sec_grp_h)

# Correlação entre sinais
corr_mat = second_group.corr()
sns.heatmap(corr_mat, annot=True)

# Correlação c/ target
sec_grp_h.corrwith(target)

# CCF das maiores correlações
# Amina flow x target
max_val, lag = plot_ccf(
    second_group.resample("min").mean().iloc[:, 1],
    target.resample("min").mean(),
    "Amina flow x target",
)
print(f"Maior correlação: {max_val} com {lag/60} horas de diferença")

# Ore pulp pH x target
max_val, lag = plot_ccf(
    second_group.resample("min").mean().iloc[:, 3],
    target.resample("min").mean(),
    "Ore pulp pH x target",
)
print(f"Maior correlação: {max_val} com {lag/60} horas de diferença")

"""
Terceiro grupo:

* Flotation column 1-7 Airflow
* Flotation column 1-7 level
"""

# Visualização das séries temporais
trd_grp_h = third_group.resample("H").mean()
mult_lineplots(trd_grp_h, 7, 2, figsize=(60, 40))

# Estacionaridade
adf_test(trd_grp_h)

# Correlação entre sinais

corr_mat = third_group.corr()
sns.heatmap(corr_mat, annot=True)
plt.figure(figsize=(20, 18))

# Correlação c/ target
third_group.corrwith(target)
# FC 01 AirFlow x target
max_val, lag = plot_ccf(
    third_group.resample("min").mean().iloc[:, 0],
    target.resample("min").mean(),
    "FC 01 AirFlow x target",
)
print(f"Maior correlação: {max_val} com {lag/60} horas de diferença")

# FC 03 AirFlow x target
max_val, lag = plot_ccf(
    third_group.resample("min").mean().iloc[:, 2],
    target.resample("min").mean(),
    "FC 03 AirFlow x target",
)
print(f"Maior correlação: {max_val} com {lag/60} horas de diferença")

# FC 05 Level x target
max_val, lag = plot_ccf(
    third_group.resample("min").mean().iloc[:, 11],
    target.resample("min").mean(),
    "FC 05 Level x target",
)
print(f"Maior correlação: {max_val} com {lag/60} horas de diferença")


"""
Quarto grupo:

* % Iron concentrate
* % Silica concentrate (target)
"""
# Visualização das séries temporais
frt_grp_h = fourth_group.resample("H").mean()
mult_lineplots(frt_grp_h, 2, 1, figsize=(30, 20))

# Estacionaridade
adf_test(frt_grp_h)

# Correlação entre sinais
corr_mat = fourth_group.corr()
sns.heatmap(corr_mat, annot=True)

# Autocorrelação do target
pd.plotting.autocorrelation_plot(clean_df.resample("min").mean().iloc[:, -1])


# Modelagem orientada a feature importance

# Revendo correlações
"""
% Silica Feed x Iron Feed (-0.97)
Flotation Column 03 Air Flow x 01 Air Flow (0.94)
01 Air Flow x 02 Air Flow (0.82)
03 Air Flow x 02 Air Flow (0.83)
"""
corrmat = clean_df.corr()
plt.figure(figsize=(30, 15))
sns.heatmap(corrmat, annot=True)
plt.tight_layout()

# Modelando com todas as features
X_train = clean_df[clean_df.index < "2017-09-02 01:00:00"].iloc[:, 0:22]
X_test = clean_df[clean_df.index >= "2017-09-02 01:00:00"].iloc[:, 0:22]
y_train = clean_df[clean_df.index < "2017-09-02 01:00:00"].iloc[:, -1]
y_test = clean_df[clean_df.index >= "2017-09-02 01:00:00"].iloc[:, -1]

y_test = pd.DataFrame(y_test)

model = XGBRegressor(n_estimators=1000)
model.fit(X_train, y_train)
y_hat = pd.DataFrame(model.predict(X_test))
y_hat = y_hat.set_index(y_test.index)

mse = mean_squared_error(y_test, y_hat)
print(f"MSE: {mse}\n RMSE: {mse**(1/2)}\n Média: {np.mean(target)}")

plot_importance(model)

# Retirando redundâncias (% Silica Feed, Flotation 02 e 03 Air Flow)
X_train = clean_df[clean_df.index < "2017-09-02 01:00:00"].iloc[:, 0:22]
X_test = clean_df[clean_df.index >= "2017-09-02 01:00:00"].iloc[:, 0:22]
y_train = clean_df[clean_df.index < "2017-09-02 01:00:00"].iloc[:, -1]
y_test = clean_df[clean_df.index >= "2017-09-02 01:00:00"].iloc[:, -1]

y_test = pd.DataFrame(y_test)

X_train = X_train.drop(
    columns=[
        "% Silica Feed",
        "Flotation Column 02 Air Flow",
        "Flotation Column 03 Air Flow",
    ]
)

X_test = X_test.drop(
    columns=[
        "% Silica Feed",
        "Flotation Column 02 Air Flow",
        "Flotation Column 03 Air Flow",
    ]
)

model = XGBRegressor(n_estimators=1000)
model.fit(X_train, y_train)
y_hat = pd.DataFrame(model.predict(X_test))
y_hat = y_hat.set_index(y_test.index)

mse = mean_squared_error(y_test, y_hat)
print(f"MSE: {mse}\n RMSE: {mse**(1/2)}\n Média: {np.mean(target)}")

plot_importance(model)
