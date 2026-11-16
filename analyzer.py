from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.cleaned_df = None
        self.stats_summary = ""
        self.anomalies_count = 0
        self.numeric_cols = []
        self.corr_matrix = None

    def load_file(self, file_path: str) -> str:
        if file_path.endswith(".csv"):
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.read_excel(file_path)

        if len(self.df) < 500 or len(self.df.columns) < 5:
            raise ValueError("Набір даних має містити не менше 500 рядків та 5 стовпців!")

        info_str = f"Успішно завантажено файл: {Path(file_path).name}\n"
        info_str += f"Кількість рядків: {len(self.df)}, стовпців: {len(self.df.columns)}\n\n"
        info_str += "--- Типи стовпців ---\n" + str(self.df.dtypes) + "\n\n"
        info_str += "--- Пропущені значення ---\n" + str(self.df.isnull().sum()) + "\n\n"
        info_str += f"Кількість дублікатів: {self.df.duplicated().sum()}\n"
        return info_str

    def process_and_analyze(self) -> str:
        if self.df is None:
            raise RuntimeError("Спочатку завантажте дані!")

        # 1. Видалення дублікатів
        self.cleaned_df = self.df.drop_duplicates().copy()

        # 2. Заповнення пропущених значень
        for col in self.cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                self.cleaned_df[col] = self.cleaned_df[col].fillna(self.cleaned_df[col].median())
            else:
                mode_val = self.cleaned_df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_val)

        # 3. Нормалізація числових даних (StandardScaler)
        self.numeric_cols = list(self.cleaned_df.select_dtypes(include=[np.number]).columns)
        if self.numeric_cols:
            scaler = StandardScaler()
            self.cleaned_df[self.numeric_cols] = scaler.fit_transform(self.cleaned_df[self.numeric_cols])

        # 4. Розрахунок статистики та кореляцій
        stats = self.cleaned_df[self.numeric_cols].describe().T
        stats["median"] = self.cleaned_df[self.numeric_cols].median()
        self.corr_matrix = self.cleaned_df[self.numeric_cols].corr()

        stats_text = "=== ОСНОВНІ СТАТИСТИЧНІ ПОКАЗНИКИ ===\n\n" + str(stats) + "\n\n"
        stats_text += "=== МАТРИЦЯ КОРЕЛЯЦІЙ ===\n\n" + str(self.corr_matrix) + "\n"
        self.stats_summary = stats_text

        # 5. Пошук аномалій (IsolationForest)
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(self.cleaned_df[self.numeric_cols])
        self.anomalies_count = int((preds == -1).sum())

        stats_text += f"\n[АНОМАЛІЇ] Знайдено аномалій методом IsolationForest: {self.anomalies_count}\n"
        return stats_text

