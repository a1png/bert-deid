from torch import nn
from transformers import Trainer

gradient_accumulation_steps = 1

class BertCRFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        neg_log_likelihood = model.neg_log_likelihood(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], inputs['labels'])
        print('likelihood:', neg_log_likelihood)
        return neg_log_likelihood
