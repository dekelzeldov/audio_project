from huggingface_hub import login
from datasets import load_dataset, Audio
from transformers import AutoModel, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
import gradio as gr
import torch


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
    self.dropout = torch.nn.Dropout(0.1) 
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
