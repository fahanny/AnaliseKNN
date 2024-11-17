# Análise Preditiva com k-Nearest Neighbors (kNN) - Influenciadores do Instagram

Este projeto utiliza o algoritmo k-Nearest Neighbors (kNN) para prever o "influence_score" de influenciadores no Instagram, com base em diversas métricas como seguidores, engajamento e curtidas. A análise visa entender os padrões de influência e otimizar a previsão do impacto de cada influenciador.

## Descrição da Base de Dados

A base de dados usada neste projeto contém informações sobre influenciadores do Instagram e suas métricas associadas:

- **rank**: Posição do influenciador no ranking.
- **channel_info**: Informações sobre o canal do influenciador.
- **influence_score**: Pontuação de influência do influenciador, que é a variável alvo para a previsão.
- **posts**: Número de posts realizados pelo influenciador.
- **followers**: Número de seguidores do influenciador.
- **avg_likes**: Número médio de curtidas por post.
- **60_day_eng_rate**: Taxa de engajamento nos últimos 60 dias.
- **new_post_avg_like**: Média de curtidas nos posts mais recentes.
- **total_likes**: Total de curtidas recebidas.
- **country**: País de origem do influenciador (convertido para continente no pré-processamento).

A coluna **country** foi mapeada para **continent**, agrupando os influenciadores por continente.

## Tecnologias Usadas

- **Python**: Linguagem principal para análise e modelagem.
- **Bibliotecas**:
  - `pandas`: Manipulação e análise de dados.
  - `numpy`: Operações matemáticas e manipulação de arrays.
  - `matplotlib` e `seaborn`: Visualização de dados.
  - `scikit-learn`: Implementação de modelos preditivos e métricas de avaliação.

## Instalação

Para rodar o projeto, siga os passos abaixo:

1. Clone o repositório:
    ```bash
    git clone https://github.com/fahanny/AnaliseKNN.git
    ```

2. Entre no diretório do projeto:
    ```bash
    cd AnaliseKNN
    ```

3. Crie um ambiente virtual (opcional, mas recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/macOS
    venv\Scripts\activate     # Para Windows
    ```

4. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Como Rodar

1. Baixe o arquivo de dados `top_insta_influencers_data.csv` e coloque-o na pasta `./data/`.

2. Execute o script Python `script.py`:
    ```bash
    python script.py
    ```

3. O código irá:

    - Carregar e limpar os dados.
    - Realizar pré-processamento, incluindo a conversão de sufixos e normalização.
    - Dividir os dados em conjuntos de treino e teste.
    - Avaliar a performance do modelo kNN utilizando validação cruzada.
    - Realizar a otimização de hiperparâmetros com GridSearchCV.
    - Avaliar o modelo no conjunto de teste com métricas como MAE, MSE e R².
    - Exibir gráficos para análise da distribuição do "influence_score" e resíduos do modelo.

## Resultados

Após a execução, você verá os seguintes resultados:

- **MAE (Erro Absoluto Médio)**: Diferença média entre os valores reais e os previstos.
- **MSE (Erro Quadrático Médio)**: Média dos quadrados dos erros.
- **R² (Coeficiente de Determinação)**: Medida da proporção de variação explicada pelo modelo.

Além disso, o código gerará gráficos para visualizar a distribuição dos dados e os resíduos do modelo.

## Contribuições

Este projeto foi desenvolvido por:

- **Fláira Hanny Bomfim dos Santos**
- **Lis Loureiro Sousa**
