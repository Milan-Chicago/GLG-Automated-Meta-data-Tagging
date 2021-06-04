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
    return list_of_lemmas
    list_of_lemmas = [word.lower() for word in spacy_doc if (word.is_stop == False) &
                      (len(word.text) > 2) &
                      (word.is_alpha) &
                      (word.pos_ in selected_POSs)]
    return list_of_lemmas


def get_top_topic(text, model_type="LDA",
                  params={"num_topics_words": 10,
                          "LDA_model_path": "./output/LDA_model",
                          "LDA_dictionary_path": "./output/dictionary.pickle"
                          }):
    """
    returns probability of the top topic and
    topic words as
    BETA1*w1 + BETA2*w2 + ... + BETA10*w10
    example:
        0.003*"nominations" + 0.002*"toys" + ... + 0.002*"toy"
    """
    if model_type == "LDA":
        lda_dictionary_path = params["LDA_dictionary_path"]
        lda_model_path = params["LDA_model_path"]
        n_topic_words = params["num_topics_words"]

        list_of_lemmas = get_list_of_lemmas(text)
        dictionary = get_LDA_dictionary(lda_dictionary_path)

        doc2bow = dictionary.doc2bow(list_of_lemmas)
        lda = get_LDA_model(lda_model_path)

        # get topic probability distribution for a document
        vector = lda[doc2bow]  # Ex.  [(4, 0.1660238), (6, 0.57233017)]
        sorted(vector, key=lambda x: x[1], reverse=True)
        topic_index, score = vector[0]

        topic_words = lda_model.print_topic(topic_index, n_topic_words)

    return topic_index, score, topic_words


################  Train LDA model ####################

def prepare_for_modeling(data_path, model_type="LDA",
                         params={"TEXT_prepared_df": pd.DataFrame({}),
                                 "save_LDA_dictionary_path": "./output/dictionary.pickle"
                                 },
                         verbose=1):
    if model_type == "LDA":
        if len(params['TEXT_prepared_df']) > 0:
            # load data for LDA
            df_data = params['TEXT_prepared_df']
            print("loaded data shape:", df_data.shape)
        elif len(data_path) > 0:
            print("Preparing data for LDA...")
            df_data = pd.read_csv(params['data_path'])
            df_data['list_of_lemmas'] = df_data['text'].apply(
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

        # select words in each article if they belong to chosen list of wordss
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


def train_model(model_type="LDA",
                params={"num_topics": 10,
                        "LDA_prepared_df": pd.DataFrame({}),
                        "LDA_dictionary_path": "./output/dictionary.pickle",
                        "save_LDA_model_path": "./output/LDA_model"
                        },
                verbose=1):
    if model_type == "LDA":
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


def name_topic(df, words_column):
    # print(df.shape)
    words_to_count = list(df[words_column])
    words_to_count = [w.upper() for l in words_to_count for w in l]
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
        df_topic[topic_column] = df_topic[topic_column].apply(str) + " " +\
            name_topic(df_topic, words_column)
        list_dfs.append(df_topic)

    return pd.concat(list_dfs)
