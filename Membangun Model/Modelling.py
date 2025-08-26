import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Membuat MLflow eksperimen baru
mlflow.set_experiment("MSML_Experiment")


# Memuat data
df = pd.read_csv("data_clean.csv")

# Mendefinisikan variabel target (y) dan fitur (X)
y = df['Urban_or_Rural']
X = df.drop('Urban_or_Rural', axis=1)

# Membagi data menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Memulai MLflow Run
try:
    with mlflow.start_run():

        # Auto log mlflow
        mlflow.sklearn.autolog(
            log_models=True,
            log_input_examples=True,
            log_model_signatures=True,
            disable=True
        )
        
        max_iter = 1000
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        # log model
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    print("\nPelatihan model selesai. Run MLflow telah dicatat.")

except Exception as e:
    print(f"Terjadi error: {e}")