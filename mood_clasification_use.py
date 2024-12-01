
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

model_id = "dekelzak/MERT-v1-95M-finetuned-DEAM_stripped_vocals"

model = utils.CustomModel(model_id)
device = "cuda:0"
model = model.to(device)

