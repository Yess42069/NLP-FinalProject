import sys

import os
import time
import math

import numpy as np
import pandas as pd

import tqdm

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW, 
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments,
    AutoModelForCausalLM
)

from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)

import datasets 
from d2l import torch as d2l


MODEL_PATH = 'roberta-large'
data_dir = "data/aclImdb"

d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')

def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    ### YOUR CODE HERE
    data = []
    labels = []
    
    dir = 'train' if is_train else 'test'
    
    for label in ['pos', 'neg']:
        # Directory path for each postive and negative
        dir_path = os.path.join(data_dir, dir, label)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                review = f.read()
                data.append(review)
                # Assign label = 1 for positive and 0 for negative
                labels.append(1 if label == 'pos' else 0)
    ### END OF YOUR CODE
    return data, labels


train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])


class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def prepare_datasets(data_dir, tokenizer, batch_size=16, max_length=512):
    train_data = read_imdb(data_dir, is_train=True)
    test_data = read_imdb(data_dir, is_train=False)

    train_dataset = IMDbDataset(train_data[0], train_data[1], tokenizer, max_length=max_length)
    test_dataset = IMDbDataset(test_data[0], test_data[1], tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, return_dict=True, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model

train_loader, test_loader = prepare_datasets(data_dir, tokenizer, batch_size=16, max_length=512)


# %%
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "key", "value"],  # Adjust as per model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"

)

peft_model = get_peft_model(model, lora_config)

# %%
# Define a function that can print the trainable parameters 
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

# %%
print(print_number_of_trainable_model_parameters(peft_model))

# %%
peft_model

# %% [markdown]
# # **Training**

# %%
def metrics(eval_prediction):
    logits, labels = eval_prediction
    pred = np.argmax(logits, axis=1)
    auc_score = roc_auc_score(labels, pred)
    return {"Val-AUC": auc_score}


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=10,
    seed=42,
    fp16=True,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
    report_to='none'
)

# Optimizer
optimizer = AdamW(peft_model.parameters(), 
                  lr=1e-4,
                  no_deprecation_warning=True)

# Scheduler
steps = (len(train_loader) * training_args.num_train_epochs) // 8
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=250)

# Trainer initialization
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=test_loader.dataset,
    tokenizer=tokenizer,
    compute_metrics = metrics,
    optimizers = (optimizer, scheduler)
)

print(f"Total Steps: 250")

# Train the model
trainer.train(resume_from_checkpoint="/kaggle/input/lora_checkpoint/transformers/default/4/checkpoint-220")

peft_model_path="/kaggle/working/peft-roberta-lora-local"

trainer.model.save_pretrained(peft_model_path) # Save the fine-tuned model
tokenizer.save_pretrained(peft_model_path) # Save the tokeni


peft_model_path="/kaggle/working/peft-roberta-lora-local"

trainer.model.save_pretrained(peft_model_path) # Save the fine-tuned model
tokenizer.save_pretrained(peft_model_path) # Save the tokeni

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

peft_model_path = "/kaggle/input/lora-output-2/transformers/default/2/peft-roberta-lora-local"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

# Load the base model and attach LoRA fine-tuning
base_model_name = "roberta-large"  # Replace with your base model
model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, peft_model_path)
model.eval()


import pandas as pd


test_file_path = "test/test_data_movie.csv"  
df = pd.read_csv(test_file_path)

print(df.head())


def preprocess_function(texts):
    return tokenizer(list(texts), truncation=True, padding="max_length", max_length=512, return_tensors="pt")

tokenized_data = preprocess_function(df["text"])


import torch

from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert to TensorDataset
dataset = TensorDataset(
    tokenized_data["input_ids"],
    tokenized_data["attention_mask"],
    torch.tensor(df["label"].values)  # Assuming the label is in the CSV
)

# Create DataLoader
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Predict
all_predictions = []
all_labels = []

model.to(device)

model.eval()
print("Test")
with torch.no_grad():
    for batch in data_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Perform inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Move predictions and labels back to CPU before appending
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


from sklearn.metrics import accuracy_score, classification_report

# Accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed metrics
report = classification_report(all_labels, all_predictions, target_names=["Negative", "Positive"])
print(report)




