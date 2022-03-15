## Rewrite text given selected features

# Preprocessing: feature extraction
To run the feature extraction, use `preprocess.py`. Write a config file following, for example, config/preprocess_dummy.yaml.

Before running the extraction, download the Fasttext word embeddings (e.g. for English), create a dir called data_auxiliary/en and put the embedding file there. Afterwards run the function write\_ranks\_into\_file in the file utils/prepare\_word\_embeddings\_frequency\_ranks.py. This should create a json file with word:rank pairs.

Afterwards, run the script:

`python preprocess.py --config configs/preproces_dummy.yaml.`

# Training a sequence-to-sequence model
Work in progress


