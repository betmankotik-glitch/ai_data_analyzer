import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from analyzer import DataAnalyzer
from ai_engine import AIReportGenerator


class AIDataAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-система аналізу та візуалізації даних")
        self.root.geometry("1100x750")

        # Підключаємо модулі аналітики та штучного інтелекту
        self.analyzer = DataAnalyzer()
        self.ai_generator = AIReportGenerator("google/flan-t5-small")

        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.LabelFrame(self.root, text=" Керування даними ", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="1. Завантажити CSV/Excel", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="2. Обробити та проаналізувати", command=self.process_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="3. Згенерувати AI-висновок", command=self.generate_ai_summary).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="4. Експорт звітів і даних", command=self.export_results).pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Вкладка 1: Статистика
        self.tab_info = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_info, text="Огляд даних і статистика")

        self.text_info = tk.Text(self.tab_info, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(self.tab_info, orient=tk.VERTICAL, command=self.text_info.yview)
        self.text_info.configure(yscrollcommand=scrollbar.set)
        self.text_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Вкладка 2: Графіки
        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Візуалізація та аномалії")

        # Вкладка 3: AI Висновок
        self.tab_ai = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ai, text="AI Висновок")

        self.text_ai = tk.Text(self.tab_ai, wrap=tk.WORD, font=("Arial", 11), padx=10, pady=10)
        ai_scrollbar = ttk.Scrollbar(self.tab_ai, orient=tk.VERTICAL, command=self.text_ai.yview)
        self.text_ai.configure(yscrollcommand=ai_scrollbar.set)
        self.text_ai.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ai_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")]
        )
        if not file_path:
            return

        try:
            info_str = self.analyzer.load_file(file_path)
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(tk.END, info_str)
            messagebox.showinfo("Успіх", "Дані успішно завантажено!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося завантажити файл:\n{e}")

    def process_data(self):
        try:
            stats_text = self.analyzer.process_and_analyze()
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(tk.END, stats_text)

            self.build_plots()
            messagebox.showinfo("Успіх", "Дані успішно оброблено та побудовано графіки!")
        except Exception as e:
            messagebox.showwarning("Попередження", str(e))

    def build_plots(self):
        for widget in self.tab_plots.winfo_children():
            widget.destroy()

        df = self.analyzer.cleaned_df
        numeric_cols = self.analyzer.numeric_cols
        corr_matrix = self.analyzer.corr_matrix

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        if len(numeric_cols) > 0:
            col1 = numeric_cols[0]
            axes[0].hist(df[col1], bins=20, color="skyblue", edgecolor="black")
            axes[0].set_title(f"Гістограма: {col1}")
            axes[0].set_xlabel("Значення")
            axes[0].set_ylabel("Частота")

            col2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            axes[1].boxplot(df[col2], patch_artist=True, boxprops=dict(facecolor="lightgreen"))
            axes[1].set_title(f"Boxplot: {col2}")

        cax = axes[2].matshow(corr_matrix, cmap="coolwarm")
        axes[2].set_title("Теплова карта кореляцій", pad=15)
        fig.colorbar(cax, ax=axes[2], shrink=0.75)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.tab_plots)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def generate_ai_summary(self):
        if not self.analyzer.stats_summary:
            messagebox.showwarning("Попередження", "Спочатку виконайте обробку даних!")
            return

        self.text_ai.delete("1.0", tk.END)
        self.text_ai.insert(tk.END, "Генерація аналітичного висновку за допомогою AI... Зачекайте.")
        self.root.update()

        try:
            report = self.ai_generator.generate_summary(
                self.analyzer.stats_summary,
                self.analyzer.anomalies_count
            )
            self.text_ai.delete("1.0", tk.END)
            self.text_ai.insert(tk.END, report)
            self.notebook.select(self.tab_ai)
        except Exception as e:
            messagebox.showerror("Помилка AI", f"Не вдалося згенерувати висновок:\n{e}")

    def export_results(self):
        if self.analyzer.cleaned_df is None:
            messagebox.showwarning("Попередження", "Немає оброблених даних для експорту!")
            return

        try:
            self.analyzer.cleaned_df.to_csv("cleaned_data.csv", index=False)
            report_text = self.text_info.get("1.0", tk.END) + "\n\n" + self.text_ai.get("1.0", tk.END)
            with open("output_report.txt", "w", encoding="utf-8") as f:
                f.write(report_text)

            messagebox.showinfo("Успіх", "Дані успішно експортовано у 'cleaned_data.csv' та 'output_report.txt'!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка під час експорту:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AIDataAnalyzerApp(root)
    root.mainloop()

