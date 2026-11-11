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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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
    """Модели и параметры для автоматического случайного поиска (задание 2)."""
    return {
        "Bagging": {
            "model": BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=SEED),
                n_jobs=-1, random_state=SEED,
            ),
            "params": {
                "n_estimators": [40, 80, 120],          # Количество деревьев
                "estimator__max_depth": [6, 10, 14],   # Глубина дерева
                "estimator__min_samples_leaf": [1, 3],
                "max_samples": [0.7, 0.9],             # Доля случайной выборки
            },
            "n_iter": 4,
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_jobs=-1, random_state=SEED),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced_subsample"],
            },
            "n_iter": 4,
        },
        "LightGBM": {
            "model": LGBMClassifier(
                class_weight="balanced", verbose=-1, n_jobs=-1, random_state=SEED,
            ),
            "params": {
                "n_estimators": [200, 350, 500],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "min_child_samples": [20, 40],
            },
            "n_iter": 6,
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                loss_function="Logloss", eval_metric="TotalF1:average=Macro",
                auto_class_weights="Balanced", verbose=False,
                thread_count=4, random_seed=SEED,
            ),
            "params": {
                "iterations": [250, 400, 550],
                "depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "l2_leaf_reg": [3, 5, 7],
            },
            "n_iter": 4,
        },
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

    # ================================================================
    # ЗАДАНИЕ 2. ОПТИМИЗАЦИЯ ЧЕРЕЗ RandomizedSearchCV
    # ================================================================
    # RandomizedSearchCV случайно выбирает комбинации параметров из
    # make_models() и сравнивает их по macro F1.
    # cv=3: три раза обучаем и проверяем каждую комбинацию на частях X_fit.
    # Все подборы выполняются ТОЛЬКО на TRAIN; X_test не используется.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    selected = {}

    for name, config in make_models().items():
        print(f"\n=== {name}: поиск гиперпараметров ===")
        if name == "CatBoost":
            # ЗАДАНИЕ 3. CatBoost имеет встроенный randomized_search().
            # Он совместим с CatBoost и ищет параметры по macro F1.
            # Используем этот метод вместо RandomizedSearchCV, чтобы
            # избежать несовместимости версий CatBoost и scikit-learn.
            best_model = config["model"]
            cat_search = best_model.randomized_search(
                param_distributions=config["params"],
                X=X_fit,
                y=y_fit,
                n_iter=config["n_iter"],
                cv=3,
                partition_random_seed=SEED,
                stratified=True,
                search_by_train_test_split=False,
                refit=True,
                verbose=False,
            )
            best_params = cat_search["params"]
            cv_f1 = max(cat_search["cv_results"]["test-TotalF1:average=Macro-mean"])
        else:
            # ЗАДАНИЕ 2. RandomizedSearchCV: случайный перебор параметров.
            search = RandomizedSearchCV(
                estimator=config["model"],            # Настраиваемый алгоритм
                param_distributions=config["params"], # Какие параметры перебираем
                n_iter=config["n_iter"],              # Число случайных комбинаций
                scoring="f1_macro",                  # Оптимизируем macro F1
                cv=cv,                               # Трёхкратная стратифицированная CV
                random_state=SEED,
                n_jobs=1,                            # Параллельность внутри моделей
                refit=True,
            )
            search.fit(X_fit, y_fit)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_f1 = search.best_score_

        # Подбираем порог классификации по macro F1 на X_val.
        threshold, val_f1 = best_threshold(best_model, X_val, y_val)
        selected[name] = {
            "model": best_model,
            "threshold": threshold,
            "val_f1": val_f1,
            "cv_f1": cv_f1,
        }
        print("Лучшие гиперпараметры:", best_params)
        print(f"Macro F1 на кросс-валидации: {cv_f1:.4f}")
        print(f"Macro F1 на валидации: {val_f1:.4f}; порог: {threshold:.2f}")

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
            "cv_macro_f1": choice["cv_f1"],
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
