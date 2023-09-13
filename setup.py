import nltk
import wget
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
wget.download('https://drive.google.com/uc?id=1E3DHBDeNSphTvflEvsFohCrBLIv4_-4G&export=download',"./data/models")