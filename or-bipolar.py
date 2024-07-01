import numpy as np

# Datos de entrenamiento para la compuerta OR con entradas y salidas bipolares
X = np.array([
    [1, -1, -1],  # Sesgo + entrada -1, -1
    [1, -1,  1],  # Sesgo + entrada -1, 1
    [1,  1, -1],  # Sesgo + entrada 1, -1
    [1,  1,  1]   # Sesgo + entrada 1, 1
])
d = np.array([-1, 1, 1, 1])  # Salida deseada bipolar

# Inicialización de pesos y tasa de aprendizaje
weights = np.zeros(X.shape[1])
learning_rate = 0.1
epochs = 100  # Número máximo de iteraciones
tolerance = 0.9  # Tolerancia para la convergencia

# Entrenamiento
for epoch in range(epochs):
    total_error = 0
    for i in range(X.shape[0]):
        y = np.dot(X[i], weights)  # Salida de ADALINE
        error = d[i] - y
        total_error += error**2
        weights += learning_rate * error * X[i]  # Actualización de pesos
    mse = total_error / X.shape[0]
    print(f'Epoch {epoch+1}, MSE: {mse:.4f}, Pesos: {weights}')
    if mse < tolerance:
        print('Convergencia alcanzada')
        break

print('Pesos finales:', weights)

# Prueba de la red entrenada
for i in range(X.shape[0]):
    y = np.dot(X[i], weights)
    print(f'Entrada: {X[i][1:]} - Salida esperada: {d[i]} - Salida ADALINE: {y:.2f}')
