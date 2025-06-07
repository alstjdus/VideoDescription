import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# 경로 설정
MODEL_NAME = "microsoft/phi-2"
# === 사용자 환경에 맞게 수정하세요 ===
OUTPUT_DIR = "path/to/save/phi2_lora_adapter"

# 여러 jsonl 파일
jsonl_files = [
    "path/to/sft/drama1.jsonl",
    "path/to/sft/drama2.jsonl",
    "path/to/sft/drama3.jsonl",
    "path/to/sft/drama4.jsonl",
    "path/to/sft/drama5.jsonl",
    "path/to/sft/drama6.jsonl",
    "path/to/sft/drama7.jsonl",
    "path/to/sft/drama8.jsonl",
    "path/to/sft/drama9.jsonl",
    "path/to/sft/drama10.jsonl",
    "path/to/sft/drama11.jsonl"
]

# Tokenizer & 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
model.config.use_cache = False

# LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# 여러 JSONL 파일을 하나의 데이터셋으로 병합
datasets = [
    load_dataset("json", data_files=file, split="train") for file in jsonl_files
]
dataset = concatenate_datasets(datasets)

# 전처리
def tokenize(example):
    full_text = f"{example['prompt']} {example['response']}"
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=384)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# 데이터 collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 학습 인자
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 학습 실행
trainer.train()

# LoRA adapter 및 tokenizer 저장
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
