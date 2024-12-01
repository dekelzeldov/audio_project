
from huggingface_hub import login
from datasets import load_dataset, Audio
from transformers import AutoModel, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
import gradio as gr
import torch
import utils

# load dataset
dataset_id = "Rehead/DEAM_stripped_vocals"
dataset = load_dataset(dataset_id, split='train[:20]') # add [:20] to loads 20 examples
dataset = dataset.train_test_split(test_size=0.1)  # Splits loaded examples into two splits: if test_size=0.2 and 20 examples: train = 16 examples, test = 4 examples
# ok so the predictions depends on the size of the test set
# currently set so test receives the exact amount or expected predictions.

##########################################################################################################################
# Encode dataset and extract features                                                                                    #
##########################################################################################################################

model_id = "m-a-p/MERT-v1-95M" 
# Loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id,trust_remote_code=True)

# Make sure the sample_rate aligned
sampling_rate = dataset["train"][0]["audio"]["sampling_rate"]
resample_rate = processor.sampling_rate
if resample_rate != sampling_rate:
    dataset = dataset.cast_column("audio", Audio(sampling_rate=resample_rate))

# Extract features
max_duration = 10.0
def preprocess_function(examples):
    device = "cuda:0"
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = processor(
        audio_arrays,
        sampling_rate=processor.sampling_rate,
        max_length=int(processor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
        padding=True
    )
    return inputs.to(device)

dataset_encoded = dataset.map(
    preprocess_function,
#    remove_columns=['audio', 'valence_std', 'valence_max_mean', 'valence_max_std', 'valence_min_mean', 'valence_min_std',   # These can be commented out since unused columns are removed
#                             'arousal_std', 'arousal_max_mean', 'arousal_max_std', 'arousal_min_mean', 'arousal_min_std'],
    batched=True,
    batch_size=100,
    num_proc=1
)

##########################################################################################################################
# 
##########################################################################################################################


model = utils.CustomModel(model_id)
device = "cuda:0"
model = model.to(device)

login()

model_name = model_id.split("/")[-1]
dataset_name = dataset_id.split("/")[-1]
path = f"{model_name}-finetuned-{dataset_name}"
batch_size = 1
gradient_accumulation_steps = 8
num_train_epochs = 1

training_args = TrainingArguments(
    path,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,

    # mem vs. preformance
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    # gradient_checkpointing=True,
    torch_empty_cache_steps=2,
    # optim="adamw_apex_fused",

    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    metric_for_best_model="r_squared",
    #use_mps_device=False,
    label_names=list(utils.label2id.keys()),
    push_to_hub=True,
    remove_unused_columns = True # with label2id working, valence_mean and arousal_mean are no longer seen as unused, so remove_unused_columns can be True or False.
)

metric = evaluate.load("r_squared")

# one of the errors occurs in this function:
# ValueError: Predictions and/or references don't match the expected format.
# Expected format: {'predictions': Value(dtype='int32', id=None), 'references': Value(dtype='int32', id=None)},
# Input predictions: [1 0],
# Input references: (array([6.4, 4.4], dtype=float32), array([6.3, 5.3], dtype=float32))
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = eval_pred.predictions
    references = np.array(eval_pred.label_ids)
    return {"r_squared": metric.compute(predictions=predictions.flatten(), references=references.flatten())}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=0, return_outputs=False):
        valance = inputs.pop("valence_mean")
        arousal = inputs.pop("arousal_mean")
        labels = torch.stack((valance, arousal),-1)

        outputs = model(**inputs) 
        logits = outputs.logits

        loss_func = torch.nn.MSELoss()
        loss = loss_func(logits, labels)

        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model,
    training_args,
    train_dataset=dataset_encoded["train"],
    eval_dataset=dataset_encoded["test"],
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer.train()
# trainer.save_model(path)


kwargs = {
    "dataset_tags": dataset_id,
    "dataset": dataset_id,
    "model_name": f"{model_name}-finetuned-{dataset_name}",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}
trainer.push_to_hub(**kwargs)

# model = AutoModel.from_pretrained(path)


