
import numpy as np

# Bibliotecas do Scikit-Learn utilizadas somente
# para carregar e preparar os dados
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CARREGAMENTO E PRE-PROCESSAMENTO
# ==========================================

# Carregar o conjunto de dados Iris
iris = load_iris()

X = iris.data
y = iris.target

# Dividir os dados em treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Padronizar os dados utilizando media 0 e desvio padrao 1
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converter os rotulos para One-Hot Encoding
num_classes = 3

y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]


# ==========================================
# 2. DEFINICAO DA ARQUITETURA DA REDE
# ==========================================

# Quantidade de neuronios em cada camada
input_size = 4
hidden_size = 8
output_size = 3

# Inicializacao aleatoria dos pesos e vieses
np.random.seed(42)

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
b2 = np.zeros((1, output_size))


# ==========================================
# 3. FUNCOES DE ATIVACAO
# ==========================================

# Funcao ReLU
def relu(Z):
    return np.maximum(0, Z)


# Derivada da funcao ReLU
def relu_derivative(Z):
    return (Z > 0).astype(float)


# Funcao Softmax
def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_values = np.exp(Z)

    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


# ==========================================
# 4. FUNCAO DE PERDA
# ==========================================

def cross_entropy(y_true, y_pred):
    epsilon = 1e-12

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    return loss


# ==========================================
# 5. TREINAMENTO DA REDE NEURAL
# ==========================================

epochs = 1000
learning_rate = 0.05

for epoch in range(epochs):

    # --------------------------------------
    # FORWARD PASS
    # --------------------------------------

    # Camada escondida
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    # Camada de saida
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Calcular a perda
    loss = cross_entropy(y_train_onehot, A2)

    # --------------------------------------
    # BACKWARD PASS (BACKPROPAGATION)
    # --------------------------------------

    m = X_train.shape[0]

    # Gradiente da camada de saida
    dZ2 = (A2 - y_train_onehot) / m

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Gradiente da camada escondida
    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # --------------------------------------
    # ATUALIZACAO DOS PESOS
    # --------------------------------------

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Mostrar o progresso do treinamento
    if (epoch + 1) % 100 == 0:
        print(f"Epoca {epoch + 1}/{epochs} - Loss: {loss:.4f}")


# ==========================================
# 6. AVALIACAO DO MODELO
# ==========================================

# Realizar a previsao com os dados de teste
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)

# Obter a classe com maior probabilidade
predictions = np.argmax(A2_test, axis=1)

# Calcular a acuracia
accuracy = np.mean(predictions == y_test)

print("\n===================================")
print("RESULTADO FINAL")
print("===================================")

print(f"Acuracia no conjunto de teste: {accuracy * 100:.2f}%")

# Mostrar algumas previsoes
print("\nComparacao entre valores reais e previstos:")

for i in range(len(y_test)):
    real = iris.target_names[y_test[i]]
    previsto = iris.target_names[predictions[i]]

    print(f"Real: {real:10s} | Previsto: {previsto}")
