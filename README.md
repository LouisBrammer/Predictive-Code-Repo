File struture explained below:

#TOTAL
main_pipeline.py -> deploys the prediction pipeline using 2 keras models and the API call to OpenAI. Currently only handels terminal text input to demo the pipeline, because whisper installations caused issues.

#TRAINING
imdb_training_template.py -> training template of models on imdb data. 
imdb_training_* -> training files of saved models

imdb_{modelname}.keras -> trained .keras models on imdb training dataset

goemotions_training_template.py -> training template of models on imdb data. 
goemotions_training_* -> training files of saved models



#TESTING
imdb_llm_test           -> measures accuracy of commercial API connection
imdb_model_comparison   -> computes the accuracy of trained models and prints out comparison plot incl. the LLM predictions


goemotions_llm_test     -> measures accuracy of commercial API connection
goemotions_model_comparison   -> computes the accuracy of trained models and prints out comparison plot incl. the LLM predictions. Is computing accuracies on different levels of aggregations.

#API call
llm_api.py 



