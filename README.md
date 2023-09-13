
## Requirements
The project has been developed and tested in a linux distribution and with Python version 3.10.

* Must install python 3.10.
* Must have pip together with virtual env installed.

### Models

To access the different models used and the datasets please contact email:
* arkhip02@ads.uni-passau.de
* daria.arkh@gmail.com. 

The data sets can not be published due to copyright restrictions on the song lyrics
For the sake of functionality we provide a small example with 10 songs. in data/processed directory.

* Google model can be found online https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
* Spacy models can be installed using the instructions from here https://spacy.io/usage/models.
  * python -m spacy download en_core_web_lg
## Installation
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 setup.py
```
## Testing
Run the command in the experiments directory
```
export  PYTHONPATH=path-to-this-directory
cd experiments
python3 bag_of_words.py 
...

```

## Structure
Divide the repository in the typical steps of the machine learning process:
Problem Definition: Clearly define the problem and goals of the machine learning project.

Data Collection: Gather the relevant data for training and evaluation.

Data Preprocessing: Clean and transform the data to make it suitable for training.

Data Splitting: Divide the dataset into training and test sets.

Model Training: Train the selected model using the training data.

Model Evaluation: Evaluate the trained model's performance using the test set.

Hyperparameter Tuning: Fine-tune the model's hyperparameters to improve performance.

- project/
    - preprocessing/
    - data/
   
        - raw/                  (Raw data collection) excluded from git use drive
        - processed/            (Preprocessed data)
        - train/                (Training data)
        - test/                 (Test data)
    - models/
        - model.py              (Model definition and training code)
    - evaluation/
        - metrics.py            (Evaluation metrics)
        - results/              (Evaluation results) starting with the name model f.e: bag of words
