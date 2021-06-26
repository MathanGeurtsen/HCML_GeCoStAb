# HCML project GeCoStAb

This repository houses the code for project GeCoStAb, an explainability comparison for tweet classification into political identity. 

The program requires the kaggle dataset to be downloaded and extracted. (also see [prerequisites](#Prerequisites)). If a shell running in the same folder as this README , then this program might be run like: 

```
python -i . "archive" "user_file.csv" yes
```

it will then perform preprocessing steps and train two models: a neural network and a decision tree. The decision tree will be visualized in `iris.pdf`, and the decision path for the user will also be shown. The neural network will also be evaluated for the user with both a SHAP and a LIME analysis. (the `-i` flag is recommended for interactivity, which keeps the server hosting the LIME and SHAP gui running) 

The models can be trained simply by running `python dnn.py` and `python decision_tree.py`, which will output training accuracy measures (this assumes the extracted kaggle dataset to be available in archive, and that preprocessing is already done). 

# Prerequisites

the required packages can be downloaded with pip: `pip install -r requirements.txt`. 
We use the nltk stopwords dataset, which needs to be downloaded as well. run the following in a python shell: 

```
import nltk
nltk.download('stopwords')
```

The model trains on a kaggle dataset (see links below). This dataset should be extracted. 

The decision tree visualization uses graphviz, which should be externally installed (e.g. by using [chocolatey](https://chocolatey.org/install): `choco install graphviz`)

# files
  - \_\_main\_\_.py : runs the entire program
  - feature_extraction: preprocessing of the data into features 
  - decision_tree: an intrinsically explainable model for tweet classification
  - dnn: a neural network for tweet classification
  - post\_proc\_explanation: a post processing explanation for the deep neural network
  - auxiliary: a file with helper functions
# links
  [kaggle: Democrats Vs Republican Tweets](https://www.kaggle.com/kapastor/democratvsrepublicantweets)  
