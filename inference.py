import numpy as np
import torch
from transformers import (
    BertForTokenClassification,
    BertTokenizer
)


class BERT_NER_inference(object):

    def __init__(self, model_path: str = "./pretrained_models/bert_ner_model.bin"):

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
        self.tag_values = checkpoint["tag_values"]

    def get_tokenizer(self):

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        return tokenizer

    def predict(self, text_input):

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
                new_labels.append(self.tag_values[label_idx])
                new_tokens.append(token)

        # just extract the named entities:
        # TODO: probably a more efficient way to do this but this is enough for now

        extracted_named_entites = []
        for entity in zip(new_tokens, new_labels):
            if entity[1] != "O":
                extracted_named_entites.append(entity)


        return {
            "extracted_entities": repr(extracted_named_entites)
        }


inference = BERT_NER_inference()

def get_model():
    return inference

print(inference.predict("Divy Murli lives in Boise, Idaho and went to college at UCSB."))









