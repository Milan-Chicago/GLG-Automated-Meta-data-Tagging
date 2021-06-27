# topic modeling libraries
from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel

# supporting libraries
import pickle
import pandas as pd
import collections
import spacy
nlp = spacy.load("en_core_web_sm")


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

################  Train LDA model ####################


def prepare_for_modeling(data_path, model_type="LDA-KeyWords",
                         params={"TEXT_prepared_df": pd.DataFrame({}),
                                 "save_LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                                 "text_column": "text"
                                 },
                         verbose=1):
    if model_type == "LDA-KeyWords":
        """
        params={"TEXT_prepared_df": pd.DataFrame({}),
                 "save_LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                 "text_column": "text"
                }
        """

        if len(params['TEXT_prepared_df']) > 0:
            # load data for LDA
            df_data = params['TEXT_prepared_df']
            if verbose == 2:
                print("loaded data shape:", df_data.shape)
        else:
            if verbose == 2:
                print("No data is provided")
            return False

        df_data['all_key_words'] = df_data['all_key_words'].apply(lambda x: [w.replace(' ', '_') for w in x
                                                                             if len(w) > 1
                                                                             ])
        # get all unique key_words
        tmp_list = df_data['all_key_words'].tolist()
        set_of_words = set([w for sublist in tmp_list for w in sublist])

        if verbose == 2:
            print('\nNumber of unique key-words for topic modeling dictionary:',
                  len(set_of_words))

        # delete empty lists of words
        df_data = df_data[df_data['all_key_words'].apply(len) > 0]

        # create a vocabulary for the LDA model
        dictionary = corpora.Dictionary(df_data['all_key_words'])

        # save dictionary
        with open(params["save_LDA_dictionary_path"], 'wb') as f:
            # Pickle the LDA dictionary using the highest protocol available.
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        if verbose == 2:
            print("LDA dictionary file is saved to:",
                  params["save_LDA_dictionary_path"])

            print('\nNumber of texts processed: ', dictionary.num_docs)
            print('Number of extracted key-words: ', len(dictionary.token2id))
            print('\nEach text is represented by list of ', len(dictionary.token2id),
                  " tuples: \n\t\t(key-words's index in bag-of-words dictionary, key-words's term frequency)")

        # count the number of occurrences of each distinct token in each document
        df_data['doc2bow'] = df_data['all_key_words'].apply(
            lambda x: dictionary.doc2bow(x))

    if model_type == "LDA":
        """
        params={"TEXT_prepared_df": pd.DataFrame({}),
                                 "save_LDA_dictionary_path": "./output/lda/dictionary.pickle",
                                 "text_column": "text"
                                 }
        """
        if len(params['TEXT_prepared_df']) > 0:
            # load data for LDA
            df_data = params['TEXT_prepared_df']
            print("loaded data shape:", df_data.shape)
        elif len(data_path) > 0:
            print("Preparing data for LDA...")
            df_data = pd.read_csv(params['data_path'])
            df_data['list_of_lemmas'] = df_data[text_column].apply(
                lambda text: get_list_of_lemmas(text))
            print("Data for LDA shape:", df_data.shape)
        else:
            return False

        # get all unique lemmas
        tmp_list = df_data['list_of_lemmas'].apply(set).apply(list).tolist()
        list_of_words = [w for sublist in tmp_list for w in sublist]

        # count words' document frequencies in the corpus
        w_freq_counter = collections.Counter(list_of_words)
        s_w_freq = pd.Series(w_freq_counter)
        if verbose == 2:
            print('\nTotal number of unique Lemmas: ', len(s_w_freq))
            print("\nDistribution of lemmas' document counts: ")
            print(pd.DataFrame(s_w_freq.describe(percentiles=[
                  0.55, 0.65, 0.75, 0.85, 0.95, 0.97, 0.99])).T)

        # select upper and lower boundary for lemmas' count
        up_pct = s_w_freq.quantile(0.99)
        low_pct = 3  # s_w_freq.quantile(0.50)
        if verbose == 2:
            print("\nDeleting too frequent and too rare words...")
            print('Lemma count upper bound:', up_pct)
            print('Lemma count lower bound:', low_pct)

        # select Lemmas
        selected_words = set(s_w_freq[(s_w_freq > low_pct)
                                      & (s_w_freq <= up_pct)].index)
        if verbose == 2:
            print('\nList of words for topic modeling dictionary is reduced from',
                  len(s_w_freq), 'to', len(selected_words))

        # select words in each article if they belong to chosen list of words
        df_data['selected_words'] = df_data['list_of_lemmas'].apply(lambda x:
                                                                    [l for l in x if l in selected_words])
        # delete empty lists of words
        df_data = df_data[df_data['selected_words'].apply(len) > 0]

        # create a vocabulary for the LDA model
        dictionary = corpora.Dictionary(df_data['selected_words'])

        # save dictionary
        with open(params["save_LDA_dictionary_path"], 'wb') as f:
            # Pickle the LDA dictionary using the highest protocol available.
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        if verbose == 2:
            print("LDA dictionary file is saved to:",
                  params["save_LDA_dictionary_path"])

            print('\nNumber of texts processed: ', dictionary.num_docs)
            print('Number of extracted lemmas: ', len(dictionary.token2id))
            print('\nEach text is represented by list of ', len(dictionary.token2id),
                  " tuples: \n\t\t(lemma's index in bag-of-words dictionary, lemma's term frequency)")

        # count the number of occurrences of each distinct token in each document
        df_data['doc2bow'] = df_data['selected_words'].apply(
            lambda x: dictionary.doc2bow(x))

    return df_data


def train_model(model_type="LDA-KeyWords",
                params={"num_topics": 10,
                        "LDA_prepared_df": pd.DataFrame({}),
                        "LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                        "save_LDA_model_path": "./output/lda_keywords/LDA_model"
                        },
                verbose=1):
    if model_type == "LDA-KeyWords":
        """
        params={"num_topics": 10,
                "LDA_prepared_df": pd.DataFrame({}),
                "LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                "save_LDA_model_path": "./output/lda_keywords/LDA_model"
                }
        """
        print("Training LDA with BERT-UMAP-HDBSCAN clustered KeyWords (NOUN_PHRASEs and VERBs)")
    if model_type == "LDA":
        """
        params={"num_topics": 10,
                "LDA_prepared_df": pd.DataFrame({}),
                "LDA_dictionary_path": "./output/lda/dictionary.pickle",
                "save_LDA_model_path": "./output/lda/LDA_model"
                }
        """
        print("Training LDA with only lemmas of NOUNs, VERBs, ADJs and ADVs")

    if len(params['LDA_prepared_df']) > 0:
        # load data for LDA
        df_data = params['LDA_prepared_df']
        if verbose == 2:
            print("loaded data shape:", df_data.shape)
    else:
        return False

    # download LDA dictionary
    dictionary = get_LDA_dictionary(params['LDA_dictionary_path'])

    # create document-term matrix for LDA
    if verbose == 2:
        print("\nCreating document-term matrix for LDA...")
    doc_term_matrix = list(df_data['doc2bow'].values)

    # define the model with chosen number of topics
    num_topics = params['num_topics']
    if verbose == 2:
        print("\nTraining LDA model with ", num_topics, " topics...")

    LDA = models.LdaMulticore
    result_lda_model = LDA(corpus=doc_term_matrix,
                           num_topics=num_topics,
                           id2word=dictionary,
                           passes=20,
                           chunksize=4000,
                           random_state=3)
    # Save model to disk
    result_lda_model.save(params["save_LDA_model_path"])
    print("LDA model file is saved to:", params["save_LDA_model_path"])

    # get topics
    df_data['infered_topics'] = df_data['doc2bow'].apply(lambda d:
                                                         sorted(result_lda_model[d],
                                                                key=lambda x: x[1],
                                                                reverse=True))
    # select top index
    df_data['top_topic'] = df_data['infered_topics'].apply(
        lambda x: x[0][0] if x[0][1] >= 0.2 else -1)
    df_data['top_topic_proba'] = df_data['infered_topics'].apply(
        lambda x: x[0][1])

    if verbose == 2:
        print(
            'Top topic indexes are selected. NOTE "-1" corresponds to top topic with probability < 20%')
    return df_data

################  Name extracted topics ####################


def name_topic(df, words_column):
    # print(df.shape)
    words_to_count = list(df[words_column])
    words_to_count = [w.replace("_", " ").lower()
                      for l in words_to_count for w in l if len(w) > 1]
    words_to_count = [w[0].upper() + w[1:] for w in words_to_count]
    # print(words_to_count[:5])

    c = collections.Counter(words_to_count)
    # print(c)
    return c.most_common(3)[1][0]


def get_topic_names(df_result, topic_column, words_column):
    list_dfs = []
    all_topics = list(set(df_result[topic_column]))
    for topic in all_topics:
        #print (topic)
        df_topic = df_result[df_result[topic_column] == topic].copy()
        #print(topic, df_topic.shape)
        df_topic[topic_column + "_name"] = name_topic(df_topic, words_column)
        list_dfs.append(df_topic)

    df_res = pd.concat(list_dfs)

    return df_res[topic_column + "_name"]

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
