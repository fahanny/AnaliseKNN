# Análise Preditiva com k-Nearest Neighbors (kNN) - Influenciadores do Instagram

## Descrição do Projeto

Este projeto utiliza o algoritmo k-Nearest Neighbors (kNN) para prever o "influence_score" de influenciadores do Instagram, com base em diversas métricas como seguidores, engajamento e curtidas. O objetivo principal é identificar os fatores que impactam a influência de um influenciador e otimizar a previsão do seu impacto com base em dados históricos. A partir dos resultados, é possível tomar decisões mais informadas sobre estratégias de marketing e colaboração com influenciadores.

## Instalação

Para rodar o projeto, siga os passos abaixo:

1. **Clone o repositório:**
    ```bash
    git clone https://github.com/fahanny/AnaliseKNN.git
    ```

2. **Entre no diretório do projeto:**
    ```bash
    cd AnaliseKNN
    ```

3. **Crie e ative um ambiente virtual (opcional, mas recomendado):**
    - Para **Linux/macOS**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    - Para **Windows**:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

4. **Instale as dependências necessárias:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

## Como Executar

Após a instalação, siga as etapas abaixo para rodar o código:

1. **Baixe o arquivo de dados** `top_insta_influencers_data.csv` e coloque-o na pasta `./data/` do projeto.

2. **Execute o script Python**:
    ```bash
    python script.py
    ```

   O código realiza as seguintes etapas:

   - Carregamento e pré-processamento dos dados, incluindo conversão de sufixos e normalização.
   - Divisão dos dados em conjuntos de treino e teste.
   - Validação cruzada para avaliar o modelo com k-Nearest Neighbors (kNN).
   - Otimização de hiperparâmetros utilizando GridSearchCV.
   - Avaliação do modelo com métricas como MAE, MSE e R².
   - Exibição de gráficos para visualização da distribuição de dados e análise de resíduos.

## Estrutura dos Arquivos

- `/data`: Contém o arquivo CSV com os dados dos influenciadores.
- `/src`: Contém o código-fonte do projeto, incluindo o script principal `script.py`.
- `/plots`: Onde os gráficos gerados são salvos.
- `README.md`: Este arquivo com as instruções do projeto.

## Tecnologias Utilizadas

- **Python**: Linguagem principal para análise e modelagem.
- **Pandas**: Manipulação e análise de dados.
- **NumPy**: Operações matemáticas e manipulação de arrays.
- **Matplotlib** e **Seaborn**: Bibliotecas para visualização de dados.
- **Scikit-learn**: Implementação de modelos preditivos e avaliação de performance.

## Autores e Colaboradores

- **Fláira Hanny Bomfim dos Santos**: Desenvolvimento do código e análise de dados.
- **Lis Loureiro Sousa**: Pré-processamento dos dados e otimização de hiperparâmetros.
