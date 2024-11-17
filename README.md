# Análise de Influência no Instagram

Este projeto tem como objetivo realizar uma análise preditiva dos influenciadores do Instagram, utilizando o algoritmo k-Nearest Neighbors (kNN) para prever o "influence_score" com base em características como número de seguidores, engajamento, e outras métricas.

## Descrição da Base de Dados

A base de dados utilizada contém informações sobre influenciadores do Instagram, incluindo:

- **rank**: Posição no ranking dos influenciadores.
- **channel_info**: Informação adicional sobre o canal do influenciador.
- **influence_score**: Pontuação de influência do influenciador.
- **posts**: Número de posts feitos pelo influenciador.
- **followers**: Número de seguidores do influenciador.
- **avg_likes**: Número médio de curtidas por post.
- **60_day_eng_rate**: Taxa de engajamento nos últimos 60 dias.
- **new_post_avg_like**: Média de curtidas nos posts mais recentes.
- **total_likes**: Total de curtidas recebidas.
- **country**: País de origem do influenciador.

Além disso, a coluna **country** foi convertida para **continent**, agrupando os influenciadores por continente.

## Instalação

Para rodar este projeto em sua máquina local, siga as etapas abaixo:

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/nome-do-repositorio.git
    ```

2. Navegue até o diretório do projeto:
    ```bash
    cd nome-do-repositorio
    ```

3. Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```

4. Certifique-se de ter os dados (`top_insta_influencers_data.csv`) na pasta `./data/` ou altere o caminho no código conforme necessário.

## Como Rodar

1. Execute o código Python no arquivo `script.py`:
    ```bash
    python script.py
    ```

2. O script irá:

    - Carregar os dados.
    - Realizar o pré-processamento (conversão de sufixos, normalização, tratamento de valores ausentes).
    - Dividir os dados em conjuntos de treino e teste.
    - Realizar validação cruzada com o modelo inicial (kNN).
    - Otimizar os hiperparâmetros utilizando GridSearchCV.
    - Avaliar o modelo no conjunto de teste, calculando métricas como MAE, MSE e R².
    - Visualizar a distribuição dos resíduos e a comparação entre valores reais e previstos.

## Resultados

Após a execução do código, os seguintes resultados serão exibidos:

- **MAE (Erro Absoluto Médio)**: Mede a diferença média absoluta entre os valores reais e previstos.
- **MSE (Erro Quadrático Médio)**: Mede a média dos quadrados dos erros.
- **R² (Coeficiente de Determinação)**: Mede a proporção da variabilidade dos dados que é explicada pelo modelo.

Além disso, gráficos serão gerados para visualizar a distribuição do "influence_score" e os resíduos do modelo.

## Contribuições

Este projeto foi desenvolvido por:

- **Fláira Hanny Bomfim dos Santos**
- **Lis Loureiro Sousa**