import pandas as pd

# Cargar el dataset
df = pd.read_csv("car data.csv")

# Ver las primeras filas del dataset
df.head()

# Ver información general sobre el dataset
df.info()

# Ver estadísticas básicas del dataset
df.describe()

# Verificar si hay valores nulos
df.isnull().sum()

# Convertir las columnas categóricas en numéricas (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# Eliminar filas con valores nulos
df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

# Definir X (variables independientes) e y (variable dependiente)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error y R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show() 
