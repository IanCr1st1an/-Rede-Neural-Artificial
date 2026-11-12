# Rede Neural Artificial - Classificação do Iris

Projeto acadêmico da disciplina de Inteligência Artificial:
implementação e treinamento, do zero, de uma rede neural artificial para
classificar as três espécies de flores do conjunto de dados Iris.

## Objetivo

Construir uma rede neural com uma camada escondida, utilizando NumPy
para os cálculos matemáticos e Scikit-Learn exclusivamente para carregar
e preparar os dados.

## Arquitetura

-   **Entrada:** 4 características do conjunto Iris.
-   **Camada escondida:** 8 neurônios com ativação ReLU.
-   **Saída:** 3 neurônios, um para cada espécie, com ativação Softmax.
-   **Treinamento:** 1.000 épocas, Cross-Entropy Loss, Backpropagation e
    Gradiente Descendente.

## Preparação dos dados

O script realiza as seguintes etapas:

1.  Carrega o conjunto Iris com `load_iris()`.
2.  Divide os dados em 80% para treinamento e 20% para teste.
3.  Padroniza as características com `StandardScaler`, ajustado somente
    nos dados de treinamento.
4.  Converte os rótulos para One-Hot Encoding.

## Tecnologias

-   Python 3
-   NumPy
-   Scikit-Learn (somente carregamento e preparação dos dados)

## Como executar

### 1. Baixe ou clone este repositório

``` bash
git clone URL_DO_REPOSITORIO
cd Rede-Neural-Artificial
```

Substitua `URL_DO_REPOSITORIO` pelo endereço deste repositório no
GitHub.

### 2. (Opcional) Crie e ative um ambiente virtual

No Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

``` bash
python -m pip install -r requirements.txt
```

### 4. Execute o programa

``` bash
python rede_neural_iris.py
```

Ao final, o programa exibe a perda durante o treinamento, a acurácia no
conjunto de teste e a comparação entre as espécies reais e previstas.

## Estrutura esperada

``` text
Rede-Neural-Artificial/
├── rede_neural_iris.py
├── requirements.txt
└── README.md
```

## Observações

-   A rede neural é implementada manualmente; não são utilizadas
    bibliotecas como TensorFlow, Keras ou PyTorch.
-   A acurácia pode variar caso sejam alterados a divisão dos dados, a
    inicialização dos pesos ou os hiperparâmetros.
-   O `random_state` e a semente do NumPy são definidos no script para
    tornar a execução reproduzível.

## Conjunto de dados

O Iris é um conjunto clássico de classificação com 150 amostras, quatro
características numéricas e três classes: Setosa, Versicolor e
Virginica. O conjunto é disponibilizado pelo Scikit-Learn.
