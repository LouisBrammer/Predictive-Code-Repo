File struture explained below:


# DEPLOYMENT
main_pipeline_whisper.py -> runs prediction models based including whisper transcription 
main_pipeline.py -> runs text-based-input prediction (was used during development)


# TRAINING 
goemotions_training_template.py
goemotions_training_trans_1.py
goemotions_training_conv2.py
goemotions_training_conv1.py

imdb_training_template.py
imdb_training_gru.py
imdb_training_conv1.py
imdb_training_conv2.py

glove.6B.50d.txt -> pretrained embedding layer used for the entire project

# TESTING

## TESTING THE FULL PIPELINE INCLUDING WHISPER
whisper_test.py -> creation of test transcription, which are manually classified and flagged if mistake happend
test_full_pipeline.py -> loads best performing sentiment model and predicts sentiments for the test transcriptions
sentiment_analysis_metrics.py -> creates metrics for full pipeline
sentiment_visualization.py -> creates plot for full pipeline metrics 

## TESTING SENTIMENTS
imdb_llm_test.py -> uses the api to predict sentiments on the test set
imdb_model_comparison.py ->  computes and compares accuracy measures of sentiment models (including trained models and LLM) 

## TESTING EMOTIONS
load_test_goemotios.py -> creates test dataset used for LLM and model comparison
goemotions_llm_test.py -> uses API call to predict emotions on test set & computes metrics for comparison
goemotions_model_comparison.py -> computes and compares accuracy measures of emotion models (including trained models and LLM)


## API CONNECTION
llm_api.py -> sets up function to use API call to predict tuple of (sentiment, emotion)

# FUNCTION HOLDER
prediction_pipeline.py -> holds one prediction function used for code clarity




