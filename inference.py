import numpy as np
import torch
from transformers import (
    BertForTokenClassification,
    BertTokenizer
)


class BERT_NER_inference(object):

    def __init__(self, model_path: str = "./pretrained_models/ner_state_dict_3.bin"):

        self.device = torch.device("cpu") # load model onto CPU for inference
        self.tokenizer = self.get_tokenizer()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = BertForTokenClassification.from_pretrained(
                        "bert-base-cased",
                        num_labels=18,
                        output_attentions=False,
                        output_hidden_states=False
                    )
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def get_tokenizer(self):

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        return tokenizer

    def predict(self, text_input):

        tag_values = ['B-org',
                      'B-tim',
                      'B-nat',
                      'B-eve',
                      'I-eve',
                      'B-per',
                      'I-tim',
                      'B-gpe',
                      'I-geo',
                      'B-art',
                      'I-gpe',
                      'I-nat',
                      'O',
                      'I-org',
                      'I-art',
                      'I-per',
                      'B-geo',
                      'PAD']

        self.model.eval()
        tokenized_input = self.tokenizer.encode(text_input)
        input_ids = torch.tensor([tokenized_input])

        with torch.no_grad():
            output = self.model.forward(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)

        return {
            "tokens": repr(new_tokens),
            "named_entities": repr(new_labels)
        }


inference = BERT_NER_inference()

def get_model():
    return inference









