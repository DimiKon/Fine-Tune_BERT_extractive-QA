
# Fine-Tune BERT on SQuAD 1.1

This repository demonstrates how to fine-tune a BERT model for a Question Answering (QA) task using the SQuAD 1.1 dataset. 

---

## Overview

- **Dataset**: Stanford Question Answering Dataset (SQuAD 1.1)
- **Model**: Pre-trained BERT (DistilBERT used as an example)
- **Goal**: Fine-tune the model to answer questions based on context paragraphs.

---

## Steps

1. **Install Dependencies**
   ```bash
   pip install datasets transformers
   ```

2. **Load and Preprocess Dataset**
   - Use `datasets` library to load SQuAD 1.1.
   - Tokenize input using `transformers.AutoTokenizer`.

3. **Fine-Tune the Model**
   - Load a pre-trained BERT model using `transformers.AutoModelForQuestionAnswering`.
   - Use the `Trainer` API for training with the following key parameters:
     - Learning rate: `2e-5`
     - Epochs: `3`
     - Batch size: `16`

4. **Evaluate and Save**
   - Evaluate the fine-tuned model on the validation set.
   - Save the trained model and tokenizer for future use.

---

## Usage

Run the following to train and evaluate the model:
```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)
trainer.train()
```

---

## Results

- Evaluate the model's performance using the validation set.
- Save the fine-tuned model:
  ```python
  model.save_pretrained("./qa_model")
  tokenizer.save_pretrained("./qa_model")
  ```

