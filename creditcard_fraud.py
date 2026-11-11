"""Лабораторная: сравнение Bagging, Random Forest, LightGBM и CatBoost по macro F1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def load_data():
    """Берём локальный creditcard.csv или скачиваем его с Kaggle."""
    csv_file = Path("creditcard.csv")
    if not csv_file.exists():
        import kagglehub

        print("Файл не найден рядом со скриптом. Скачиваем датасет с Kaggle...")
        folder = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_file = Path(folder) / "creditcard.csv"

    return pd.read_csv(csv_file)


def best_threshold(model, X_val, y_val):
    """Находим порог вероятности, дающий лучший macro F1 на валидации."""
    probabilities = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0

    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_val, predictions, average="macro")
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)

    return best_t, best_f1


def make_models():
    """Для каждого алгоритма: обычный и настроенный варианты."""
    return {
        "Bagging": [
            BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=8, random_state=SEED),
                n_estimators=40, max_samples=0.8, n_jobs=-1, random_state=SEED,
            ),
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=SEED
                ),
                n_estimators=80, max_samples=0.8, n_jobs=-1, random_state=SEED,
            ),
        ],
        "Random Forest": [
            RandomForestClassifier(
                n_estimators=120, max_depth=10, n_jobs=-1, random_state=SEED,
            ),
            RandomForestClassifier(
                n_estimators=220, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
            ),
        ],
        "LightGBM": [
            LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                class_weight="balanced", verbose=-1, n_jobs=-1, random_state=SEED,
            ),
            LGBMClassifier(
                n_estimators=400, learning_rate=0.03, num_leaves=31,
                min_child_samples=30, reg_lambda=1,
                class_weight="balanced", verbose=-1, n_jobs=-1, random_state=SEED,
            ),
        ],
        "CatBoost": [
            CatBoostClassifier(
                iterations=250, depth=5, learning_rate=0.08,
                auto_class_weights="Balanced", verbose=False,
                thread_count=4, random_seed=SEED,
            ),
            CatBoostClassifier(
                iterations=450, depth=6, learning_rate=0.04,
                l2_leaf_reg=5, auto_class_weights="Balanced",
                verbose=False, thread_count=4, random_seed=SEED,
            ),
        ],
    }


def main():
    # 1. Загрузка данных; Class — целевой столбец (0 — норма, 1 — мошенничество).
    df = load_data()
    X = df.drop(columns="Class")
    y = df["Class"]
    print("Всего транзакций:", len(df))
    print("Распределение классов:\n", y.value_counts())

    # 2. Разбиение по условию: тест 30%, с той же долей классов.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=df.Class, random_state=SEED,
    )

    # Ещё 20% от train оставляем для подбора параметров и порога.
    # Тестовая выборка в подборе НЕ участвует.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED,
    )

    # 3. Сравниваем два набора параметров каждого метода на валидации.
    selected = {}
    for name, variants in make_models().items():
        print(f"\n=== {name} ===")
        for number, model in enumerate(variants, 1):
            model.fit(X_fit, y_fit)
            threshold, val_f1 = best_threshold(model, X_val, y_val)
            print(f"Вариант {number}: macro F1={val_f1:.4f}, порог={threshold:.2f}")

            if name not in selected or val_f1 > selected[name]["val_f1"]:
                selected[name] = {
                    "model": model,
                    "threshold": threshold,
                    "val_f1": val_f1,
                }

    # 4. Переобучаем лучшие варианты на всех 70% train и оцениваем на test.
    results = []
    predictions_by_model = {}
    for name, choice in selected.items():
        model = clone(choice["model"])
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= choice["threshold"]).astype(int)
        score = f1_score(y_test, predictions, average="macro")

        results.append({
            "model": name,
            "validation_macro_f1": choice["val_f1"],
            "test_macro_f1": score,
            "threshold": choice["threshold"],
        })
        predictions_by_model[name] = predictions

    # 5. Таблица результатов; лучшая по тесту модель — для матрицы ошибок.
    table = pd.DataFrame(results).sort_values("test_macro_f1", ascending=False)
    print("\nРЕЗУЛЬТАТЫ НА TEST (macro F1):")
    print(table.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
    table.to_csv("results.csv", index=False)

    best_name = table.iloc[0]["model"]
    best_predictions = predictions_by_model[best_name]
    print(f"\nЛучшая модель на тесте: {best_name}")
    print(classification_report(y_test, best_predictions, digits=4))

    # 6. Матрица ошибок — для метода с максимальным macro F1 на тесте.
    matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    print("Матрица ошибок:\n", matrix)
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["Legal", "Fraud"]
    ).plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion matrix: {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
