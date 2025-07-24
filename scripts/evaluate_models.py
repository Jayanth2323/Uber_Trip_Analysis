# scripts/evaluate_models.py

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def evaluate_and_plot(y_true, predictions: dict, output_dir="plots"):
    import os

    os.makedirs(output_dir, exist_ok=True)

    for model_name, y_pred in predictions.items():
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"🔍 {model_name.upper()} — MAPE: {mape:.2%}, R²: {r2:.4f}")

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(y_true.values, label="Actual", color="black")
        plt.plot(y_pred, label=f"{model_name} Predicted", linestyle="--")
        plt.title(f"{model_name.upper()} — Actual vs Predicted")
        plt.xlabel("Time Step")
        plt.ylabel("Trips")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_prediction.png")
        plt.close()
