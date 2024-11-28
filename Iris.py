import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Iris.csv')
df = df.drop(columns=['Id'])
X = df.drop(columns=['Species'])
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_knn_model(p):
    knn = KNeighborsClassifier(n_neighbors=5, p=p)
    knn.fit(X_train, y_train)
    return knn

knn_models = {
    "Khoảng cách Euclidean": train_knn_model(2),
    "Khoảng cách Manhattan": train_knn_model(1),
    "Khoảng cách Minkowski": train_knn_model(3),
    "Khoảng cách Chebyshev": train_knn_model(float('inf'))
}

new_data = pd.DataFrame([[3.8, 3.4, 1.6, 0.2]], columns=X.columns)
new_data_scaled = scaler.transform(new_data)


def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác của mô hình {name}: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    prediction = model.predict(new_data_scaled)
    print(f"Loài hoa dự đoán với mô hình {name}: {prediction[0]}")

    distances, indices = model.kneighbors(new_data_scaled)
    print(f"\nKhoảng cách {name} đến các điểm gần nhất:", distances)
    print(f"Chỉ số của các điểm gần nhất:", indices)
    print("\n")

    plot_learning_curve(model, f'Learning Curve: {name}', X, y)


def plot_learning_curve(model, title, X, y):
    plt.figure()
    plt.title(title)
    plt.xlabel('Số lượng ví dụ trong tập huấn luyện')
    plt.ylabel('Điểm số')

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Score")

    plt.legend(loc="best")
    plt.grid()
    plt.show()

for name, model in knn_models.items():
    evaluate_model(model, name)
