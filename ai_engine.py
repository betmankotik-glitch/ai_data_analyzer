import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AIReportGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"[INFO] Завантаження локальної AI моделі ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate_summary(self, stats_summary: str, anomalies_count: int) -> str:
        messages = [
            {"role": "system", "content": "Ти професійний бізнес-аналітик. Пиши стисло, грамотно, українською мовою."},
            {"role": "user", "content": f"Зроби короткий аналітичний висновок (2-3 речення) по набору даних на 550 рядків, де знайдено {anomalies_count} аномалій (викидів)."}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.6
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        ai_result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        full_report = f"=== AI АНАЛІТИЧНИЙ ЗВІТ ===\n\n{ai_result}\n\n"
        full_report += f"Загальна кількість знайдених аномалій у даних (IsolationForest): {anomalies_count}\n"
        return full_report