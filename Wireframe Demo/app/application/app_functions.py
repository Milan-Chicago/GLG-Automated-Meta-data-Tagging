import re
# topic modeling libraries
from gensim import models, corpora

# supporting libraries
import pickle
import pandas as pd
import collections
import numpy as np
from typing import Dict

import spacy
nlp = spacy.load("en_core_web_sm")

import torch
from pydantic import BaseModel
from transformers import (
    BertForTokenClassification,
    BertTokenizer
)

################  NER using pretrained model ####################


class BERT_NER_inference(object):
  """
  This class is meant to load a pretrained BERT NER model from a saved
  checkpoint onto the CPU to be used for inference
  """

  def __init__(self, model_path):

    self.device = torch.device("cpu")  # load model onto CPU for inference
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

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased', do_lower_case=False)

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

    tokens = self.tokenizer.convert_ids_to_tokens(
        input_ids.to('cpu').numpy()[0])
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

################  Topics for new texts using pretrained model ####################


def get_LDA_model(path):
  lda_model = models.LdaMulticore.load(path)
  return lda_model


def get_LDA_dictionary(path):
  with open(path, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    dictionary = pickle.load(f)
  return dictionary


def get_list_of_lemmas(text):
  # extract 'NOUN', 'VERB', 'ADJ', 'ADV' from text
  # if they are not stop-words, have length>2 and have only alphabetic characters
  selected_POSs = ['NOUN', 'VERB', 'ADJ', 'ADV']

  spacy_doc = nlp(text)
  list_of_lemmas = [word.text.lower() for word in spacy_doc if (word.is_stop == False) &
                    (len(word.text) > 2) &
                    (word.is_alpha) &
                    (word.pos_ in selected_POSs)]
  return list_of_lemmas


def get_top_topic_index(text,
                        params={"LDA_dictionary_path": "./output/lda/dictionary1.pickle",
                                "LDA_model_path": "./output/lda/LDA_model1"
                                }
                        ):
  list_of_lemmas = get_list_of_lemmas(text)

  # load topic dictionary
  with open(params['LDA_dictionary_path'], 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    dictionary = pickle.load(f)

  doc2bow = dictionary.doc2bow(list_of_lemmas)

  # load topic model
  lda = get_LDA_model(params['LDA_model_path'])

  # get (topic,proba) tuples
  vector = lda[doc2bow]

  topic_number, proba = sorted(vector, key=lambda item: item[1])[-1]

  if proba < 0.2:
    return -1, -1
  else:
    return topic_number, proba


################  Process unseen text ####################


def predict_topics(text,
                   params={"topics_df_path": './output/lda/topics.pickle',
                           "first_dictionary_path": "./output/lda/dictionary1.pickle",
                           "first_LDA_model_path": "./output/lda/LDA_model1"
                           }
                   ):

  # load pre-trained topics
  LDA_topics_df_path = params["topics_df_path"]
  with open(LDA_topics_df_path, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    df_topics = pickle.load(f)
  df_topics.head(1)

  # first level
  first_LDA_dict_path = params["first_dictionary_path"]
  first_LDA_model_path = params["first_LDA_model_path"]
  t1, t1_proba = get_top_topic_index(text,
                                     params={"LDA_dictionary_path": first_LDA_dict_path,
                                             "LDA_model_path": first_LDA_model_path
                                             }
                                     )

  # second level
  second_LDA_dict_path = first_LDA_dict_path[:-
                                             7] + "_" + str(t1 + 1) + ".pickle"
  second_LDA_model_path = first_LDA_model_path + "_" + str(t1 + 1)
  t2, t2_proba = get_top_topic_index(text,
                                     params={"LDA_dictionary_path": second_LDA_dict_path,
                                             "LDA_model_path": second_LDA_model_path
                                             }
                                     )

  # third level
  third_LDA_dict_path = first_LDA_dict_path[:-7] + \
      "_" + str(t1 + 1) + "_" + str(t2 + 1) + ".pickle"
  third_LDA_model_path = first_LDA_model_path + \
      "_" + str(t1 + 1) + "_" + str(t2 + 1)
  t3, t3_proba = get_top_topic_index(text,
                                     params={"LDA_dictionary_path": third_LDA_dict_path,
                                             "LDA_model_path": third_LDA_model_path
                                             }
                                     )

  # get topic names
  t1_name = df_topics[df_topics['first_level_topic']
                      == t1]['first_level_topic_name'].iloc[0]
  t2_name = df_topics[df_topics['second_level_topic'] == str(t1) +
                      '.' + str(t2)]['second_level_topic_name'].iloc[0]
  t3_name = df_topics[df_topics['third_level_topic'] == str(t1) +
                      '.' + str(t2) + '.' + str(t3)]['third_level_topic_name'].iloc[0]

  dict_output = {'first_level_topic': t1,
                 'first_level_topic_name': t1_name,
                 'first_level_topic_proba': t1_proba,
                 'second_level_topic': t2,
                 'second_level_topic_name': t2_name,
                 'second_level_topic_proba': t2_proba,
                 'third_level_topic': t3,
                 'third_level_topic_name': t3_name,
                 'third_level_topic_proba': t3_proba
                 }
  return dict_output
