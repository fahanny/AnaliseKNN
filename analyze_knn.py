import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Ignorar warnings desnecessários
warnings.filterwarnings("ignore")

# Função para converter sufixos em valores numéricos
def convert_suffixed_columns(df, columns):
    replace = {'b': 'e9', 'm': 'e6', 'k': 'e3', '%': ''}
    df[columns] = df[columns].replace(replace, regex=True).astype(float)
    return df

# Função para visualizar distribuição
def plot_distribution(data, column, title):
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], kde=True, bins=20, color="teal", edgecolor="black")
    plt.title(f"{title} - Distribuição de {column}")
    plt.xlabel(column)
    plt.ylabel("Frequência")
    plt.show()

# Função para visualização de resíduos
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    
    # Histograma dos resíduos
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=20, color="purple", edgecolor="black")
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.title("Distribuição dos Resíduos")
    plt.xlabel("Resíduos")
    plt.ylabel("Frequência")

    # Real vs Previsto com cores diferenciadas
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.7, c='darkorange', label="Valores Previstos")
    plt.scatter(y_test, y_test, alpha=0.7, c='steelblue', label="Valores Reais")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Linha de Perfeição")
    plt.title("Real vs. Previsto")
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Previstos")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Carregamento dos dados
data_path = "./data/top_insta_influencers_data.csv"
data = pd.read_csv(data_path)

# Seleção de colunas
columns_to_keep = ['rank', 'channel_info', 'influence_score', 'posts', 'followers', 'avg_likes', 
                   '60_day_eng_rate', 'new_post_avg_like', 'total_likes', 'country']
data = data[columns_to_keep]

# Mapeamento de 'country' para 'continent'
continent_map = {
    'Argentina': 'South America', 'Brazil': 'South America', 'USA': 'North America',
    'United States': 'North America', 'Canada': 'North America', 'Spain': 'Europe',
    'Netherlands': 'Europe', 'United Kingdom': 'Europe', 'Uruguay': 'South America',
    'Turkey': 'Asia', 'Indonesia': 'Asia', 'Colombia': 'South America', 'France': 'Europe',
    'Australia': 'Oceania', 'Italy': 'Europe', 'United Arab Emirates': 'Asia', 'Puerto Rico': 'North America',
    "CÃ´te d'Ivoire": 'Africa', 'Germany': 'Europe', 'India': 'Asia', 'Anguilla': 'North America',
    'Switzerland': 'Europe', 'Sweden': 'Europe', 'British Virgin Islands': 'North America',
    'Czech Republic': 'Europe', 'Mexico': 'North America', 'Russia': 'Asia'
}
data['continent'] = data['country'].map(continent_map)
data['continent'].fillna('Unknown', inplace=True)

# Conversão de colunas com sufixos
columns_to_convert = ['total_likes', 'posts', 'followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like']
data = convert_suffixed_columns(data, columns_to_convert)

# Tratamento de valores inválidos
data[columns_to_convert] = data[columns_to_convert].replace([np.inf, -np.inf], np.nan)
data.dropna(subset=columns_to_convert + ['continent'], inplace=True)

# Normalização
scaler = StandardScaler()
data[columns_to_convert] = scaler.fit_transform(data[columns_to_convert])

# Visualização da distribuição da variável-alvo
plot_distribution(data, 'influence_score', "Distribuição de Influence Score")

# Divisão em treino e teste
X = data[columns_to_convert]
y = data['influence_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Validação cruzada e cálculo de MSE
knn = KNeighborsRegressor()
cv_scores_mse = cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"MSE Médio (Cross-Validation): {-np.mean(cv_scores_mse):.2f}")

# Busca de hiperparâmetros
param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Melhores Hiperparâmetros
best_knn = grid_search.best_estimator_
print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")

# Avaliação no conjunto de teste
y_pred = best_knn.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.3f}")

# Visualização dos resíduos
plot_residuals(y_test, y_pred)
