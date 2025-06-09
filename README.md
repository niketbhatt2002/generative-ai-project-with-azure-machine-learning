# Fine-Tuning T5-small for Text Summarization.

This project fine-tunes the `t5-small` transformer model on the **XSum** dataset for text summarization tasks.  
It involves loading a dataset, preprocessing, training the model, saving the fine-tuned model, and evaluating its performance.

---

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Saving the Model](#saving-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Credits](#credits)

---

## Installation

First, install the required libraries:

```bash
pip install transformers datasets
```

---

## Dataset

We use the **XSum** dataset, a collection of BBC articles paired with single-sentence summaries.

```python
from datasets import load_dataset
dataset = load_dataset("xsum")
```

You can also use your own custom dataset by modifying the loading step.

---

## Model

We fine-tune the **T5-small** model for summarization:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

---

## Preprocessing

Documents are tokenized, truncated, and padded:

```python
def preprocess_function(examples):
    return tokenizer(examples["document"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

---

## Training

The model is trained using Hugging Face's `Trainer` API with the following parameters:

- **Learning rate**: `2e-5`
- **Batch size**: `8`
- **Epochs**: `2`
- **Weight decay**: `0.01`
- **Save and evaluate** at the end of each epoch

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(100)),  # Using 100 samples
    eval_dataset=tokenized_dataset["validation"].select(range(20)),  # Using 20 samples
)

trainer.train()
```

---

## Saving the Model

After training, the model and tokenizer are saved for future use:

```python
model.save_pretrained("./fine-tuned-t5-small-summarization")
tokenizer.save_pretrained("./fine-tuned-t5-small-summarization")
```

---

## Evaluation

The model is evaluated on the test dataset using the **ROUGE** metric:

```python
from transformers import pipeline
from datasets import load_metric

summarizer = pipeline("summarization", model="./fine-tuned-t5-small-summarization")
sample = dataset["test"][0]
summary = summarizer(sample["document"], max_length=50, min_length=10, do_sample=False)[0]["summary_text"]

rouge = load_metric("rouge")
reference = sample["summary"]
scores = rouge.compute(predictions=[summary], references=[reference])

print("ROUGE Scores:", scores)
```

---

## Results

The ROUGE scores printed will give an indication of the summarization model's performance based on precision, recall, and F1 metrics.

---

## Credits

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- BBC XSum dataset

---
