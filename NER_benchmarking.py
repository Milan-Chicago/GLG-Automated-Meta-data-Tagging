import pandas as pd
import numpy as np
import spacy
import collections

NER_data = pd.read_csv("./data/ner_dataset.csv")
nlp = spacy.load("en_core_web_sm")


print(NER_data[24:54])


"""
doc = nlp(sentence)

# Display the entities found by the model, and the type of each.
print('{:<12}  {:}\n'.format('Entity', 'Type'))

# For each entity found...
for i, ent in enumerate(doc.ents):
    # Print the entity text `ent.text` and its label `ent.label_`.
    print('{:<12}  {:} {:} {:}'.format(ent.text, ent.label_, ent.start, ent.end))

print(NER_data.head())

NER_non_na = NER_data.notna()
sentence_positions = list(NER_non_na["Sentence #"])
sentence_positions = np.argwhere(sentence_positions)
sentence_positions = sentence_positions.reshape((sentence_positions.shape[0], ))
print(sentence_positions[:10])
print(len(sentence_positions))

print(sentence_positions[4:6])
"""

def extract_sentence_positions(NER_data):

    NER_non_na = NER_data.notna()
    document_positions = list(NER_non_na["Sentence #"])
    document_positions = np.argwhere(document_positions)
    document_positions = document_positions.reshape((document_positions.shape[0],))

    return document_positions

def make_aggregate_predictions_spacy(NER_model, num_docs, document_positions):

    ner_mapping_dict = {
        "GPE": "geo",
        "PERSON": "per",
        "TIME": "tim",
        "EVENT": "eve"
    }

    all_sentences = list(NER_data["Word"])
    all_named_entities = list(NER_data["Tag"])

    document_positions = document_positions[:(num_docs + 1)]
    predicted_named_entities = []
    ground_truth_named_entities = []
    sentences = []

    gpe_ground_truths = []
    for i in range(len(document_positions) - 1):

        doc_position = document_positions[i]
        # print(doc_position)
        sentence = ' '.join(all_sentences[document_positions[i]:document_positions[i+1]])
        #print("length of ground truth doc: {}".format(document_positions[i+1]-document_positions[i]))
        #print(sentence)
        gt_named_entities = list(zip(all_sentences[document_positions[i]:document_positions[i+1]], all_named_entities[document_positions[i]:document_positions[i+1]]))
        pred_named_entities = []

        sentences.append(sentence)
        doc = NER_model(sentence)

        if len(doc) == document_positions[i+1]-document_positions[i]:
            for token in doc:
                pred_named_entities.append((token, token.ent_iob_, token.ent_type_))

                if token.ent_type_ == "GPE":
                    assert token.text == all_sentences[doc_position]
                    #if token.text == all_sentences[doc_position]:
                    gpe_ground_truths.append((token, all_sentences[doc_position], all_named_entities[doc_position], '-'.join((token.ent_iob_, token.ent_type_))))

                doc_position += 1

        else:
            doc_position += 1
            continue

        predicted_named_entities.append(pred_named_entities)
        ground_truth_named_entities.append(gt_named_entities)

    ground_truth_distribution = list(zip(*gpe_ground_truths))[2]


    return sentences, predicted_named_entities, ground_truth_named_entities, gpe_ground_truths, collections.Counter(ground_truth_distribution)

document_positions = extract_sentence_positions(NER_data)
print(document_positions[:10])

sentences, predicted_named_entities, ground_truth_named_entities, gpe_ground_truths, retrieved_ground_truths = make_aggregate_predictions_spacy(nlp, 100, document_positions)

#print(sentences)
#print(predicted_named_entities[0])
#print(ground_truth_named_entities[0])
print(len(gpe_ground_truths))
print(retrieved_ground_truths)





