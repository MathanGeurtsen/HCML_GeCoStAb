# HCML project GeCoStAb

This repository houses the code for project GeCoStAb, an explainability comparison for tweet classification into political identity. 

The models can be run simply by running `python dnn.py` and `python decision_tree.py`

# Prerequisites

the required packages can be downloaded with pip: `pip install -r requirements.txt`. 
We use the nltk stopwords dataset, which needs to be downloaded as well. run the following in a python shell: 

```import nltk
nltk.download('stopwords')```

# files
  - feature_Extraction: preprocessing of the data into features 
  - decision_tree: an intrinsically explainable model for tweet classification
  - dnn: a deep neural network for tweet classification
  - post\_proc\_explanation: a post processing explanation for the deep neural network
  - auxiliary: a file with helper functions
# links
  [paper on overleaf](https://www.overleaf.com/project/60af9ea97d79e4b596362f91) 
  
  [drive folder](https://drive.google.com/drive/folders/1xhBWr8afTJo8ohb0zpWb6nJVgSg4-A4U)
  
