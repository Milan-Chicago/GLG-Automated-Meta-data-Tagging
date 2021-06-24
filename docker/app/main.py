import numpy as np
import torch
from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from transformers import (
    BertForTokenClassification,
    BertTokenizer
)

class BERT_NER_inference(object):
    """
    This class is meant to load a pretrained BERT NER model from a saved
    checkpoint onto the CPU to be used for inference
    """

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

        """
        Specify the tokenizer, only supports regular BERT for now
        TODO: Add flexibility for DistilBERT and MobileBERT
        """

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        return tokenizer

    def predict(self, text_input):

        """
        Perform inference, using a pretrained model.
        :param text_input (str): feed in a string input
        :return: dict of named tokens and named entites
        TODO: Add model prediction probabilities
        """

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

        return {
            "tokens": repr(new_tokens),
            "named_entities": repr(new_labels)
        }


inference = BERT_NER_inference()

def get_model():
    return inference

app = FastAPI()


class NER_Request(BaseModel):

    text: str


class NER_Response(BaseModel):

    predictions: Dict[str, str]


@app.post("/predict", response_model=NER_Response)
def predict(request: NER_Request, model: BERT_NER_inference = Depends(get_model)):

    predictions = model.predict(request.text)

    return NER_Response(predictions=predictions)