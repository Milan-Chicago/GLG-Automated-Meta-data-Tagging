from flask import Flask, render_template, request, redirect, Markup, flash, session, url_for, redirect
from application import app
from application.forms import TextForm
from werkzeug.utils import secure_filename

import os
import pandas as pd
from application import app_functions as af

########################## Test and manage dictionary files  ##########################


@ app.route('/', methods=['POST', 'GET'])
def home():
  return render_template('home.html')


@ app.route('/process_text', methods=['POST', 'GET'])
def process_text():
  basedir = os.path.abspath(os.path.dirname(__file__))
  try:
    form = TextForm()

    if form.validate_on_submit():
      text = form.text.data

      # process text
      nar_model_path = os.path.join(basedir, 'ner_models/bert_ner_model.bin')
      nar_dict = af.get_all_named_entities(text, nar_model_path)

      topics_dict = af.predict_topics(text,
                                      params={"topics_df_path": os.path.join(basedir, 'lda_models/lda_keywords/topics.pickle'),
                                              "first_dictionary_path": os.path.join(basedir, "lda_models/lda_keywords/dictionary1.pickle"),
                                              "first_LDA_model_path": os.path.join(basedir, "lda_models/lda_keywords/LDA_model1")
                                              }
                                      )

      output = {"text": text,
                "topic_1": topics_dict['first_level_topic_name'],
                "topic_1_proba": round(topics_dict['first_level_topic_proba'] * 100, 2),
                "topic_2": topics_dict['second_level_topic_name'],
                "topic_2_proba": round(topics_dict['second_level_topic_proba'] * 100, 2),
                "topic_3": topics_dict['third_level_topic_name'],
                "topic_3_proba": round(topics_dict['third_level_topic_proba'] * 100, 2),
                "key_words": nar_dict
                }

      return render_template('text_processed_output.html', output=output)
    return render_template('text_upload.html', form=form)

  except Exception as e:
    flash('Failed to process text: ' + str(e), 'danger')
    return render_template('text_upload.html', form=form)
