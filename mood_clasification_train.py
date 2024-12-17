

from datasets import load_dataset, Audio
from transformers import AutoModel, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
import torch
from sklearn.model_selection import KFold

# >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# >>> y = np.array([1, 2, 3, 4])
# >>> kf = KFold(n_splits=2)
# >>> kf.get_n_splits(X)
# 2
# >>> print(kf)
# KFold(n_splits=2, random_state=None, shuffle=False)
# >>> for i, (train_index, test_index) in enumerate(kf.split(X)):
# ...     print(f"Fold {i}:")
# ...     print(f"  Train: index={train_index}")
# ...     print(f"  Test:  index={test_index}")
# Fold 0:
#   Train: index=[2 3]
#   Test:  index=[0 1]
# Fold 1:
#   Train: index=[0 1]
#   Test:  index=[2 3]

DATA = False

if DATA:
    # load dataset
    dataset_id = "Rehead/DEAM_stripped_vocals"
    dataset = load_dataset(dataset_id, split='train+test') # add [:20] to loads 20 examples
    dataset = dataset.train_test_split(test_size=0.1, seed=24)  # Splits loaded examples into two splits: if test_size=0.2 and 20 examples: train = 16 examples, test = 4 examples


    from datasets import load_dataset

    # First make the kfold object
    kf = KFold(n_splits=10, shuffle=True)
    num_k_fold_ephocs = 3



##########################################################################################################################
# Encode dataset and extract features                                                                                    #
##########################################################################################################################

model_id = "m-a-p/MERT-v1-95M" 
# Loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)

if DATA:
    # Make sure the sample_rate aligned
    sampling_rate = dataset["train"][0]["audio"]["sampling_rate"]
    resample_rate = processor.sampling_rate
    if resample_rate != sampling_rate:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=resample_rate))

# Extract features
max_duration = 45
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

if DATA:
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

id2label = {
    "0": "valence_mean", 
    "1": "arousal_mean"
}
label2id = {v: k for k, v in id2label.items()}

class CustomModel(torch.nn.Module):
  def __init__(self, checkpoint): 
    super(CustomModel,self).__init__() 

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            problem_type = "regression",
            num_labels = len(label2id),
            label2id=label2id,
            id2label=id2label,
            #device_map="auto",
            output_scores = True
        )
    self.dropout = torch.nn.Dropout(0.5) 
    self.classifier = torch.nn.Linear(768, len(label2id)) # load and initialize weights

  def forward(self, input_values=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_values=input_values, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None: # labels is none so this is never hit
      loss_fct = torch.nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


model = CustomModel(model_id)
device = "cuda:0"
model = model.to(device)

model_name = model_id.split("/")[-1]
if DATA:
    dataset_name = dataset_id.split("/")[-1]
    path = f"{model_name}-finetuned-{dataset_name}"
else:
    path = f"{model_name}-finetuned"
batch_size = 1
gradient_accumulation_steps = 8
num_train_epochs = 1

training_args = TrainingArguments(
    path,
    eval_strategy="epoch",
    save_strategy="no",
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
    label_names=list(label2id.keys()),
    # push_to_hub=True,
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
    references = np.array(eval_pred.label_ids)/5-1
    references = np.transpose(references)
    print(f"references: {references} \t predictions: {predictions}")
    return {"r_squared": metric.compute(predictions=predictions.flatten(), references=references.flatten())}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=0, return_outputs=False):
        valance = inputs.pop("valence_mean")
        arousal = inputs.pop("arousal_mean")
        labels = torch.stack((valance, arousal),-1)
        labels = labels/5-1

        outputs = model(**inputs) 
        logits = outputs.logits

        loss_func = torch.nn.MSELoss()
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss

if DATA:
    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["test"],
        processing_class=processor,
        compute_metrics=compute_metrics,
    )


LOAD = True
TRAIN = False
SAVE = False


file_name = f'model_weights_norm.pth'

if(LOAD):
    print()
    print(f"loading from {file_name}")
    model.load_state_dict(torch.load(file_name, weights_only=True))

if(TRAIN):
    print()
    
    print("training")
    model.train()
    splits = kf.split(dataset_encoded["train"])
    for i, (train_idxs, val_idxs) in enumerate(splits): 
        print(f"Fold #: {i}")
        fold_dataset = DatasetDict({
            "train": dataset_encoded["train"].select(train_idxs),
            "validation": dataset_encoded["train"].select(val_idxs),
        })
        trainer.train_dataset = fold_dataset["train"]
        trainer.eval_dataset = fold_dataset["validation"]
        trainer.train()


if(SAVE):
    print()
    print(f"saving to {file_name}")
    torch.save(model.state_dict(), file_name)
  
if DATA:
    trainer.eval_dataset = dataset_encoded["test"]
    print("testing")
    model.eval()
    print(trainer.evaluate())

print()
print()
print("**********************")
print("evaluating custom data")
print("**********************")

from scipy.io import wavfile
import torchaudio.transforms as T
import os

model.eval()

# assign directory
directory = '.\presentation_songs'

# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)

        input_sr, input_array = wavfile.read(f)

        resample_rate = processor.sampling_rate
        # make sure the sample_rate aligned
        if resample_rate != input_sr:
            print(f'setting rate from {input_sr} to {resample_rate}')
            resampler = T.Resample(input_sr, resample_rate)
        else:
            resampler = None

        # audio file is decoded on the fly
        if resampler:
            input_array = resampler(torch.from_numpy(input_array.astype(dtype='f4')))

        if input_array.size()[-1] == 2:
            # convert sterio to mono by summing
            input_array = input_array.sum(axis=-1)

        device = "cuda:0"
        inputs = processor(
            input_array,
            sampling_rate=processor.sampling_rate,
            max_length=int(processor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            padding=True
        )
        inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"\tvalance: {outputs.logits[0][0]}, arousal:{outputs.logits[0][1]}")

