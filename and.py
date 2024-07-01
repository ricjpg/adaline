import numpy as np

# Datos de entrenamiento para la compuerta AND con salida bipolar
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
d = np.array([-1, -1, -1, 1])  # Salida deseada bipolar

# Inicialización de pesos y tasa de aprendizaje
weights = np.zeros(X.shape[1])
learning_rate = 0.1
epochs = 20  # Número de iteraciones

# Función de activación bipolar
def activation_function(y):
    return 1 if y >= 0 else -1

# Entrenamiento
for epoch in range(epochs):
    for i in range(X.shape[0]):
        y = np.dot(X[i], weights)  # Salida de ADALINE
        y_activated = activation_function(y)
        error = d[i] - y_activated
        weights += learning_rate * error * X[i]  # Actualización de pesos
    print(f'Epoch {epoch+1}, Pesos: {weights}')

print('Pesos finales:', weights)

# Prueba de la red entrenada
for i in range(X.shape[0]):
    y = np.dot(X[i], weights)
    y_activated = activation_function(y)
    print(f'Entrada: {X[i][1:]} - Salida esperada: {d[i]} - Salida ADALINE: {y_activated}')
