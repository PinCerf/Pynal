"""Лабораторная: сравнение Bagging, Random Forest, LightGBM и CatBoost по macro F1."""

from pathlib import Path
from io import StringIO

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

# Названия моделей и гиперпараметров для понятного вывода.
MODEL_NAMES = {
    "Bagging": "Бэггинг (Bagging)",
    "Random Forest": "Случайный лес (Random Forest)",
    "LightGBM": "Градиентный бустинг LightGBM",
    "CatBoost": "Градиентный бустинг CatBoost",
}

PARAM_NAMES = {
    "n_estimators": "Количество деревьев",
    "iterations": "Количество итераций",
    "max_depth": "Максимальная глубина дерева",
    "estimator__max_depth": "Максимальная глубина базового дерева",
    "depth": "Глубина дерева",
    "min_samples_leaf": "Минимум объектов в листе",
    "estimator__min_samples_leaf": "Минимум объектов в листе базового дерева",
    "max_samples": "Доля обучающих объектов для дерева",
    "class_weight": "Балансировка классов",
    "num_leaves": "Максимум листьев дерева",
    "learning_rate": "Скорость обучения",
    "min_child_samples": "Минимум объектов в листе",
    "l2_leaf_reg": "L2-регуляризация",
}


def explain_param(key, value):
    """Переводим название и значение гиперпараметра на понятный русский."""
    title = PARAM_NAMES.get(key, key)
    if value is None and "depth" in key:
        value = "без ограничения"
    elif value is None and key == "class_weight":
        value = "без дополнительных весов"
    elif value == "balanced_subsample":
        value = "балансировка в каждом дереве"
    return f"  {title}: {value}"




def load_data():
    """Берём локальный creditcard.csv или скачиваем его с Kaggle."""
    csv_file = Path("creditcard.csv")
    if not csv_file.exists():
        import kagglehub

        print("Исходный файл creditcard.csv не найден. Загружаем данные с Kaggle...")
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
    print("\n" + "=" * 68)
    print("1. ИСХОДНЫЕ ДАННЫЕ")
    print("=" * 68)
    print(f"Всего транзакций: {len(df):,}".replace(",", " "))
    print(f"Легальных транзакций (класс 0): {(y == 0).sum():,}".replace(",", " "))
    print(f"Мошеннических транзакций (класс 1): {(y == 1).sum():,}".replace(",", " "))
    print(f"Доля мошеннических: {(y == 1).mean() * 100:.3f}%")
    print("Класс 0 — обычная транзакция, класс 1 — мошенничество.")

    # 2. Разбиение по условию: тест 30%, с той же долей классов.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=df.Class, random_state=SEED,
    )

    # Ещё 20% от train оставляем для подбора параметров и порога.
    # Тестовая выборка в подборе НЕ участвует.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED,
    )

    print("\n" + "=" * 68)
    print("2. РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 68)
    print(f"Обучающая выборка (70%): {len(X_train)} транзакций.")
    print(f"Тестовая выборка (30%): {len(X_test)} транзакций.")
    print(f"Для обучения моделей: {len(X_fit)} транзакций.")
    print(f"Для подбора порога: {len(X_val)} транзакций.")
    print("Пропорции классов сохранены (stratify).")
    print("Тестовая выборка НЕ используется при подборе параметров.")

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
        print("\n" + "-" * 68)
        print(f"3. НАСТРОЙКА МОДЕЛИ: {MODEL_NAMES[name]}")
        print("-" * 68)
        print(f"Проверяем {config['n_iter']} случайных наборов параметров.")
        print("Для каждого набора выполняется трёхкратная кросс-валидация.")
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
                log_cout=StringIO(),  # Скрываем служебные англоязычные логи CatBoost.
                log_cerr=StringIO(),
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
        print("Лучшие найденные параметры:")
        for key, value in best_params.items():
            print(explain_param(key, value))
        print(f"Macro F1 при кросс-валидации: {cv_f1:.4f}")
        print("  Это среднее качество на внутренних проверках при подборе параметров.")
        print(f"Macro F1 на валидации: {val_f1:.4f}")
        print("  Это качество на отдельной проверочной части обучающих данных.")
        print(f"Лучший порог мошенничества: {threshold:.2f}")
        print("  Если оценка вероятности класса 1 не ниже порога, модель ставит класс 1.")

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

    # ================================================================
    # ЗАДАНИЕ 2. РЕЗУЛЬТАТЫ НА ТЕСТЕ И СРАВНЕНИЕ АЛГОРИТМОВ
    # ================================================================
    table = pd.DataFrame(results).sort_values("test_macro_f1", ascending=False)

    readable = table.rename(columns={
        "model": "Алгоритм",
        "cv_macro_f1": "F1: кросс-валидация",
        "validation_macro_f1": "F1: валидация",
        "test_macro_f1": "F1: тест",
        "threshold": "Порог",
    }).copy()
    readable["Алгоритм"] = readable["Алгоритм"].map(MODEL_NAMES)

    print("\\n" + "=" * 68)
    print("4. СРАВНЕНИЕ АЛГОРИТМОВ ПО MACRO F1")
    print("=" * 68)
    print(readable.to_string(index=False, float_format=lambda number: f"{number:.4f}"))
    print("\\nРасшифровка таблицы:")
    print("  F1: кросс-валидация — качество при подборе гиперпараметров.")
    print("  F1: валидация — качество при подборе порога классификации.")
    print("  F1: тест — главная итоговая оценка на новых данных.")
    print("  Порог — начиная с какой оценки вероятности ставится класс 1.")
    print("  Чем ближе Macro F1 к 1, тем лучше классификация.")
    readable.to_csv("results.csv", index=False, sep=";", encoding="utf-8-sig")
    print("Таблица сохранена: results.csv")

    # Лучший алгоритм определяется по полному значению F1 на тесте.
    best_name = table.iloc[0]["model"]
    best_predictions = predictions_by_model[best_name]
    report = classification_report(
        y_test, best_predictions, output_dict=True, zero_division=0
    )

    print("\\n" + "=" * 68)
    print("5. ЛУЧШАЯ МОДЕЛЬ И ЕЁ МЕТРИКИ")
    print("=" * 68)
    print(f"Лучший алгоритм: {MODEL_NAMES[best_name]}")
    print(f"Macro F1 на тесте: {table.iloc[0]['test_macro_f1']:.4f}")
    print("Это среднее F1 двух классов: обычных и мошеннических операций.")

    print("\\nПоказатели по каждому классу:")
    print(f"{'Класс':<24} {'Precision':>11} {'Recall':>9} {'F1':>9} {'Кол-во':>9}")
    for key, title in [("0", "Легальные (0)"), ("1", "Мошеннические (1)")]:
        row = report[key]
        print(
            f"{title:<24} {row['precision']:>11.4f} "
            f"{row['recall']:>9.4f} {row['f1-score']:>9.4f} "
            f"{int(row['support']):>9}"
        )

    print("\\nЧто означают метрики:")
    print("  Precision — доля правильных срабатываний среди предсказаний класса.")
    print("  Recall — доля найденных объектов среди всех реальных объектов класса.")
    print("  F1 — общая оценка точности и полноты.")
    print("  Кол-во — сколько объектов этого класса было в тестовой выборке.")
    print(f"  Accuracy (общая доля правильных ответов): {report['accuracy']:.4f}")
    print(f"  Macro F1 (равный вес обоих классов): {report['macro avg']['f1-score']:.4f}")
    print("  При редком мошенничестве Macro F1 информативнее одной лишь Accuracy.")

    # ================================================================
    # ЗАДАНИЕ 2. МАТРИЦА ОШИБОК ЛУЧШЕЙ МОДЕЛИ
    # ================================================================
    matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    print("\\n" + "=" * 68)
    print("6. МАТРИЦА ОШИБОК ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 68)
    print("                       Предсказано 0    Предсказано 1")
    print(f"Реально 0 (легальная)      {tn:>10}       {fp:>10}")
    print(f"Реально 1 (мошенничество)  {fn:>10}       {tp:>10}")
    print(f"  {tn} — верно распознаны легальные транзакции (TN).")
    print(f"  {fp} — легальные ошибочно названы мошенническими (FP).")
    print(f"  {fn} — мошеннические транзакции пропущены (FN).")
    print(f"  {tp} — мошеннические транзакции найдены правильно (TP).")
    print(f"  Всего мошеннических в тесте: {tp + fn}.")
    print(f"  Обнаружено мошеннических: {tp} из {tp + fn} "
          f"({tp / (tp + fn) * 100:.2f}%)." if (tp + fn) else
          "  В тесте нет мошеннических транзакций.")

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Легальная (0)", "Мошенническая (1)"],
    ).plot(values_format="d", cmap="Blues")
    plt.title(f"Матрица ошибок: {MODEL_NAMES[best_name]}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Настоящий класс")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Матрица ошибок сохранена: confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
