import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# M2U5 Крок 1. Завантажте датасет
df = pd.read_csv("game_data.csv")
print(df.head())
print(df.info())

# Крок 2. Перевірте дані
print("Кількість рядків:", len(df))
print("Колонки:", list(df.columns))
print("\nПорожні значення:")
print(df.isnull().sum())

target_column = "action"

print("\nКількість прикладів кожної дії:")
print(df[target_column].value_counts())

# Крок 3. Виберіть ознаки (відповідно до вашого collect_data.py)
feature_columns = [
    "player_x",
    "obstacle_x",
    "obstacle_y",
    "distance_x",
    "distance_y"
]

# Крок 4. Створіть X і y
X = df[feature_columns]
y = df[target_column]
print(X.head())
print(y.head())

# Крок 5. Розділіть дані
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Крок 6. Створіть модель
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

# Крок 7. Навчіть модель
model.fit(X_train, y_train)

joblib.dump(model, "game_model.pkl")
print("Модель успішно збережено у файл game_model.pkl")

# Крок 8. Отримайте прогнози
predictions = model.predict(X_test)
print("Перші прогнози:")
print(predictions[:10])

# Крок 9. Перевірте точність
accuracy = accuracy_score(y_test, predictions)
print("Точність моделі:", round(accuracy, 2))

# Крок 10. Порівняйте прогнози з реальними діями
results = pd.DataFrame({
    "real_action": y_test,
    "predicted_action": predictions,
})
print(results.head(10))

comparison = pd.crosstab(
    results["real_action"],
    results["predicted_action"],
)
print("\nТаблиця порівняння (crosstab):")
print(comparison)