"""
THIS IS A STAND-ALONE SCRIPT MEANT TO BE RUN BEFORE RUNNING train.py
This script takes command line arguments.

Open train, val and test files for the seq2seq model.
Extract features and write these files in data_preprocessed/

"""

# read in config from configs/preprocess.yaml

# for split in splits:
# open file and yield lines (in utils/helpers.py)
# extract features, round the features to 2 decimal points (if last one is 0, then 1), string
# write the sentences with prepended control tokens into files in data_preprocessed/
