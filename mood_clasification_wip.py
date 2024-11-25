
from huggingface_hub import notebook_login
from datasets import load_dataset, Audio
from transformers import AutoModel, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
import gradio as gr
from torch import nn

# load demo audio and set processor
dataset_id = "Rehead/DEAM_stripped_vocals"
# dataset = load_dataset(dataset_id)
dataset = load_dataset(dataset_id, split='train[:20]')
dataset = dataset.train_test_split(test_size=0.2)


# def generate_audio():
#     example = dataset["train"].shuffle()[0]
#     audio = example["audio"]
#     return (
#         audio["sampling_rate"],
#         audio["array"],
#     ), (example["valence_mean"], example["arousal_mean"])


# with gr.Blocks() as demo:
#     with gr.Column():
#         for _ in range(4):
#             audio, label = generate_audio()
#             output = gr.Audio(audio, label=label)

# demo.launch(debug=True)



model_id = "m-a-p/MERT-v1-95M"
# loading our model weights
#model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id,trust_remote_code=True)

# make sure the sample_rate aligned
sampling_rate = dataset["train"][0]["audio"]["sampling_rate"]
resample_rate = processor.sampling_rate
if resample_rate != sampling_rate:
    dataset = dataset.cast_column("audio", Audio(sampling_rate=resample_rate))

max_duration = 45.0
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
    remove_columns=['audio', 'valence_std', 'valence_max_mean', 'valence_max_std', 'valence_min_mean', 'valence_min_std',
                             'arousal_std', 'arousal_max_mean', 'arousal_max_std', 'arousal_min_mean', 'arousal_min_std'],
    batched=True,
    batch_size=100,
    num_proc=1
)

id2label = {
    "0": "valence_mean", 
    "1": "arousal_mean"
}
label2id = {v: k for k, v in id2label.items()}

class CustomModel(nn.Module):
  def __init__(self,checkpoint,num_labels): 
    super(CustomModel,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            problem_type = "regression",
            num_labels = 2,
            label2id=label2id,
            id2label=id2label,
            #device_map="auto",
            output_scores = True
        )
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(768, num_labels) # load and initialize weights

  def forward(self, input_values=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_values=input_values, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)

model = CustomModel(model_id, 2)
device = "cuda:0"
model = model.to(device)

# notebook_login()

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 2 #10


training_args = TrainingArguments(
    f"{model_name}-finetuned-gtzan",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    use_mps_device=False,
    # label_names=list(label2id.keys())
    # push_to_hub=True,
    remove_unused_columns = False
)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        valance = inputs.pop("valence_mean")
        arousal = inputs.pop("arousal_mean")
        outputs = model(**inputs) # Here the exception happens
        labels = valance + arousal
        logits = outputs.logits
        loss_func = nn.MSELoss()
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

# kwargs = {
#     "dataset_tags": dataset_id,
#     "dataset": "DEAM_stripped_vocals",
#     "model_name": f"{model_name}-finetuned-DEAM_stripped_vocals",
#     "finetuned_from": model_id,
#     "tasks": "audio-classification",
# }
# trainer.push_to_hub(**kwargs)