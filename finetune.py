import datasets
import torch
from transformers import AutoTokenizer,GPT2LMHeadModel
from peft import get_peft_model, LoraConfig

from ftdp.trainer import DPTrainer
from ftdp.data import DataCollatorForPrivateCausalLanguageModeling
from ftdp.args import PrivacyArguments
from ftpd.args import DPTrainingArguments as TrainingArguments
from dp_transformers.dp_utils import OpacusDPTrainer
#Load Model & Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = "[PAD]" #Adds padding for max length consistency while tokenization

#Load dataset using HF's Dataset
dataset = datasets.load_dataset('reddit', split="train[:10000]").train_test_split(0.02,seed=44)

print("=="*50)
print(dataset.keys())
#Tokenization
dataset = dataset.map(
    lambda batch: tokenizer(batch['content'],padding="max_length",truncation=True, max_length=128),
    batched=True, remove_columns=dataset.column_names['train']
)
print("=="*50)
print(dataset.keys())

#initiate peft_config
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.0,
    r=4 #lora_dim
)

model = get_peft_model(model=model, peft_config=peft_config)

model.train()

data_collator = DataCollatorForPrivateCausalLanguageModeling(tokenizer)
# Set up some privacy_args
privacy_args = PrivacyArguments(
    per_sample_max_grad_norm=1.0,
    noise_multiplier=None,
    target_epsilon=8.0,
    target_delta=None,
    disable_dp=False
)

#Set up the Training Arguments
train_args = TrainingArguments(
    output_dir="logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.0003,
    num_train_epochs=2,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4
)

# Initialize DPTrainer
trainer = DPTrainer(
    model = model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator,
    privacy_args=privacy_args,
    args=train_args
)
try:
    trainer.train()
finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        print({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })