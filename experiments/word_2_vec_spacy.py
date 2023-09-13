from evaluation.Word2Vec import Word2VecSpacy

spacy_model_experiment = Word2VecSpacy("../data/processed/limit-10.json")
spacy_model_experiment.execute()
spacy_model_experiment.print_evaluation()
print(spacy_model_experiment.result_df.head())
