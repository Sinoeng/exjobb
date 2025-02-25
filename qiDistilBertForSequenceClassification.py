from iDistilbert import *

import torch
import torch.nn as nn
import torch.optim as optim

import transformers
from transformers import DistilBertModel, DistilBertForMaskedLM, DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification, DistilBertConfig, DataCollatorWithPadding

class qiDistilBertForSequenceClassification(iDistilBertForSequenceClassification):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    **kwargs
    ):
        x = self.quantize(input_ids)
        output = super(input_ids = x, **kwargs)
        return self.dequantize(output)

teacher_id = "distilbert/distilbert-base-uncased"
teacher_config = DistilBertConfig(    
    distance_metric = "cosine_distance",
    activation_function = "softmax",
    signed_inhibitor =  False,
    alpha = 0,
    center = False,
    output_contexts = True,
    output_hidden_states = False,
)
    
qmodel = qiDistilBertForSequenceClassification.from_pretrained(
        teacher_id,
        config=teacher_config,
    )

print(qmodel)