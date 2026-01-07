# Complete Executable Notebook Code

Since the notebooks currently only have skeleton code, here's the complete, copy-paste ready code for each notebook. You can either:

**Option A:** Copy-paste this code directly into Colab cells
**Option B:** I'll update the notebook files with complete code

---

## Notebook 02: Data Exploration - COMPLETE CODE

### Use this in Google Colab:

Open a new Colab notebook and copy these cells:

```python
# CELL 1: Setup
import sys
sys.path.append('/content/llm-finetuning-production')

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
print("✓ Setup complete")
```

```python
# CELL 2: Load Dolly Dataset
print("Loading Dolly-15k...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
print(f"Size: {len(dolly):,} examples")
print(f"Columns: {dolly.column_names}")
print(f"\nFirst example:\n{dolly[0]}")
```

```python
# CELL 3: Analyze Categories
categories = Counter(dolly['category'])

plt.figure(figsize=(14, 6))
plt.bar(categories.keys(), categories.values())
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Dolly-15k Categories')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

for cat, count in categories.most_common():
    print(f"{cat}: {count:,} ({count/len(dolly)*100:.1f}%)")
```

```python
# CELL 4: Text Length Analysis
instruction_lens = [len(t.split()) for t in dolly['instruction']]
response_lens = [len(t.split()) for t in dolly['response']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.hist(instruction_lens, bins=50, edgecolor='black')
ax1.set_title(f'Instructions (Mean: {np.mean(instruction_lens):.1f} words)')
ax1.set_xlabel('Word Count')

ax2.hist(response_lens, bins=50, edgecolor='black', color='green')
ax2.set_title(f'Responses (Mean: {np.mean(response_lens):.1f} words)')
ax2.set_xlabel('Word Count')

plt.tight_layout()
plt.show()
```

```python
# CELL 5: Load Classification Dataset
print("Loading AG News...")
ag_news = load_dataset("SetFit/ag_news", split="train")
print(f"Size: {len(ag_news):,}")
print(f"\nFirst example:\n{ag_news[0]}")
```

```python
# CELL 6: Class Distribution
labels = ['World', 'Sports', 'Business', 'Sci/Tech']
label_counts = Counter(ag_news['label'])

plt.figure(figsize=(10, 6))
plt.bar([labels[i] for i in sorted(label_counts.keys())],
        [label_counts[i] for i in sorted(label_counts.keys())])
plt.title('AG News Class Distribution')
plt.ylabel('Count')
plt.show()
```

```python
# CELL 7: Tokenization Analysis
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Sample and tokenize
sample = dolly[0]['instruction'] + " " + dolly[0]['response']
tokens = tokenizer.encode(sample)

print(f"Text: {sample[:100]}...")
print(f"\nTokens ({len(tokens)}): {tokens[:20]}...")

# Analyze token counts
dolly_sample = dolly.select(range(1000))
token_counts = [
    len(tokenizer.encode(ex['instruction'] + ' ' + ex['response']))
    for ex in dolly_sample
]

plt.hist(token_counts, bins=50)
plt.axvline(512, color='red', linestyle='--', label='512 limit')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.title(f'Token Distribution (Mean: {np.mean(token_counts):.1f})')
plt.legend()
plt.show()

print(f"\nExamples > 512 tokens: {sum(1 for x in token_counts if x > 512)/len(token_counts)*100:.1f}%")
```

```python
# CELL 8: Create Splits
dolly_split = dolly.train_test_split(test_size=0.1, seed=42)
ag_split = ag_news.train_test_split(test_size=0.1, seed=42)

print("Dolly split:")
print(f"  Train: {len(dolly_split['train']):,}")
print(f"  Val: {len(dolly_split['test']):,}")

print("\nAG News split:")
print(f"  Train: {len(ag_split['train']):,}")
print(f"  Val: {len(ag_split['test']):,}")

print("\n✓ Data exploration complete!")
```

---

## Notebook 03: Baseline Evaluation - COMPLETE CODE

```python
# CELL 1: Setup
import sys
sys.path.append('/content/llm-finetuning-production')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

print("✓ Imports complete")
```

```python
# CELL 2: Load GPT-2 for Generation
print("Loading GPT-2...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

tokenizer.pad_token = tokenizer.eos_token

print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

```python
# CELL 3: Test Generation
test_prompts = [
    "The capital of France is",
    "To make a cake, you need",
    "Python is a programming language that"
]

print("Zero-shot generation:")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    generated = tokenizer.decode(outputs[0])
    print(f"\nPrompt: {prompt}")
    print(f"Output: {generated}")
```

```python
# CELL 4: Compute Perplexity on Sample Data
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
sample_texts = [ex['response'] for ex in dolly.select(range(100))]

model.eval()
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for text in tqdm(sample_texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        outputs = model(**encodings, labels=encodings["input_ids"])

        total_loss += outputs.loss.item() * encodings["input_ids"].size(1)
        total_tokens += encodings["input_ids"].size(1)

perplexity = np.exp(total_loss / total_tokens)
print(f"\nBaseline GPT-2 Perplexity: {perplexity:.2f}")

# Clean up
del model
torch.cuda.empty_cache()
```

```python
# CELL 5: Load BERT for Classification
from transformers import pipeline

print("Loading BERT for zero-shot classification...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# Test on AG News
ag_news = load_dataset("SetFit/ag_news", split="test[:100]")
candidate_labels = ["world news", "sports", "business", "science and technology"]

predictions = []
for ex in tqdm(ag_news.select(range(20)), desc="Classifying"):
    result = classifier(ex['text'], candidate_labels)
    pred_label = candidate_labels.index(result['labels'][0])
    predictions.append(pred_label)

# Compute accuracy
true_labels = [ex['label'] for ex in ag_news.select(range(20))]
accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)

print(f"\nZero-shot classification accuracy: {accuracy*100:.1f}%")
```

```python
# CELL 6: Summary
print("=" * 60)
print("BASELINE EVALUATION SUMMARY")
print("=" * 60)
print(f"\n✓ GPT-2 Perplexity: {perplexity:.2f}")
print(f"✓ Zero-shot Classification: {accuracy*100:.1f}%")
print("\n✓ Baselines established!")
print("  Next: Notebook 04 - Full Finetuning")
print("=" * 60)
```

---

## Notebook 04: Full Finetuning GPT-2 - COMPLETE CODE

```python
# CELL 1: Setup
import sys
sys.path.append('/content/llm-finetuning-production')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from src.utils.memory import print_gpu_utilization

print("✓ Setup complete")
print_gpu_utilization()
```

```python
# CELL 2: Load Model and Data
print("Loading GPT-2 and TinyStories...")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Load small dataset for quick training
dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
```

```python
# CELL 3: Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("✓ Data tokenized")
```

```python
# CELL 4: Configure Training
output_dir = "/content/drive/MyDrive/llm_finetuning_checkpoints/gpt2_tinystories"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision
    report_to="none",  # Disable wandb for now
    save_total_limit=2
)

print("✓ Training configured")
```

```python
# CELL 5: Train!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

print("Starting training...")
print_gpu_utilization()

trainer.train()

print("\n✓ Training complete!")
```

```python
# CELL 6: Test Generation
model.eval()

test_prompts = [
    "Once upon a time",
    "The little girl",
    "In a magical forest"
]

print("\nGeneration samples:")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
    print(f"\nPrompt: {prompt}")
    print(f"Story: {tokenizer.decode(outputs[0])}")
```

```python
# CELL 7: Save Model
save_path = "/content/drive/MyDrive/llm_models/gpt2_tinystories_finetuned"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✓ Model saved to {save_path}")
print("✓ Full finetuning complete!")
```

---

##QUICK START GUIDE

**To use these:**

1. Open Google Colab
2. Create a new notebook
3. Copy-paste the cells for the notebook you want
4. Run cells sequentially
5. Each notebook is complete and executable!

**Or I can update the actual notebook files with this code - would you prefer that?**

Let me know and I'll either:
- A) Update all notebook files with complete code
- B) Provide more complete code for specific notebooks
- C) Create a single "quick start" notebook with all the essentials

What would work best for you?
