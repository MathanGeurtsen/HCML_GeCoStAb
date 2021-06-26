# HCML project GeCoStAb

This repository houses the code for project GeCoStAb, an explainability comparison for tweet classification into political identity. 

The program requires the kaggle dataset to be downloaded, and requires the path to the extracted folder as an argument, together with a path to a user_file it is to explain (also see [prerequisites](#Prerequisites)). If a shell running in the same folder as this README , then this program might be run like: 

```
python . "archive" "user_file.csv" yes
```

it will then perform preprocessing steps and train two models: a neural network and a decision tree. The decision tree will visualized in `decision_tree_viz`, the neural network will be evaluated locally on the `user_file`. 

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
  - feature_Extraction: preprocessing of the data into features 
  - decision_tree: an intrinsically explainable model for tweet classification
  - dnn: a deep neural network for tweet classification
  - post\_proc\_explanation: a post processing explanation for the deep neural network
  - auxiliary: a file with helper functions
# links
  [kaggle: Democrats Vs Republican Tweets](https://www.kaggle.com/kapastor/democratvsrepublicantweets)
  [paper on overleaf](https://www.overleaf.com/project/60af9ea97d79e4b596362f91) 
  
  [drive folder](https://drive.google.com/drive/folders/1xhBWr8afTJo8ohb0zpWb6nJVgSg4-A4U)
  
