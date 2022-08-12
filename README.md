# Rewrite text given selected features

This codebase is a re-implementation of the [ACCESS](https://arxiv.org/abs/1910.02677) model for controllable text generation using control tokens of four features:
- dependency length
- word frequency
- length (number of characters)
- Levenshtein similarity ratio

The model used is a Transformer implemented with `fairseq`. 
A prerequisite of training is extensive preprocessing with feature extraction.

## Requirements
Install a conda environment for python 3.6.

Install the requirements from `requirements.txt` with pip using `pip install -r requirements.txt`.
If there is an issue with spacy language models, comment those lines out, try installing with pip again.

Afterwards, download the language models separately using
- `python -m spacy download en_core_web_sm` for English
- `python -m spacy download de_core_news_sm` for German

Other dependencies: `easse` (install following the [instructions](https://github.com/feralvam/easse)), `tensorboardX`(with pip), `nltk` stopwords (`python -m nltk.downloader stopwords`).


Download the Fasttext embeddings from this [site](https://fasttext.cc/docs/en/crawl-vectors.html) and put the unzipped file into the directory:
- `data_auxiliary/en` for English
- `data_auxiliary/de` for German

The unpreprocessed training, validation and test sets should be put in `data/en` or `data/de` with file names: `train.[src|tgt]`, `valid.[src|tgt]` and `test.[src|tgt]`.

In order to use the 'sentencepiece' tokenizer (as in the original ACCESS code) you need to place a trained [SentencePiece tokenizer](https://github.com/google/sentencepiece) in the 'data_auxiliary/[en|de]' folder. We use the pretrained sentencepiece tokenizers from [here](https://github.com/facebookresearch/cc_net): Download their trained LM for English / German following the instructions in the readme (section "Training Language Models") and place the downloaded '[en|de].sp.model' file in the 'data_auxiliary/[en|de]' folder (you do not need the '[en|de].arpa.bin' file).

## Preprocessing: feature extraction
The preprocessing involves several steps:
1) Prepare word frequency ranks: <br>
Run the function `write_ranks_into_file` from the script in `utils/prepare_word_embeddings_frequency_ranks.py`. This should create a json file in `data_auxiliary/en` (or `/de`) with `word:rank` pairs.

2) Prepare a configuration file for preprocessing: <br>
The configuration file should follow the example in `configs/preprocess_dummy.yaml` <br>
Arguments in the preprocessing config:
   - `experiment_id`: ignore this
   - `lang`: the language of the dataset, either `en` or `de` supported for now
   - `features`: a list of features to be extracted from the dataset. The options are dependency, frequency, length, levenshtein. 
   - `analyze_features`: a functionality that plots histograms for the feature values. Boolean.
   - `overwrite`: ignore this
It's best to run the preprocessing with feature extraction for all four features once because it is time-consuming. The features that won't be used in the experiments can be removed from files after.
   
3) Prepare output directory: <br>
Create the directory where the newly processed data will be stored given the requested features (and the parent directories if needed). The directory name should be the name of the extracted features in alphabetical order, separated by '_'. This directory should be created at the following place: `[repository_directory]/data_preprocessed/[lang]`.  For example, use `mkdir` to create `data_preprocessed/en/frequency_length` if you plan to preprocess English data and extract frequency and length features.

4) Preprocess corpus: <br>
Run the preprocessing script, `python preprocess.py --config configs/[config_file].yaml --tokenizer [tokenizer_type] --shard [shard_name]` where
   - `config_file`: name of the configuration file
   - `tokenizer_type`: optional, can be either 'spacy' (for using the spacy lm for tokenization) or 'sentpiece' (for sentence piece tokenization); default is 'sentpiece'
   - `shard_name`: optional, can be used to process only a part of the corpus instead of processing the target and source files of the train, test and validation split together, see the [Wiki](https://github.com/coli-saar/rewrite_text/wiki/Optional-Scripts-Preprocessing) for more information about the usage.

Depending on the dataset size and the features this step might take some time, so it's best to run it with `nohup` or `screen`.

Once finished, the preprocessed files should be located in a subdirectory of `data/en` or `data/de`. For example, if the features [dependency, frequency] were selected, the preprocessed files will be in `dependency_frequency/`.

## Training a sequence-to-sequence model
For training first prepare the configuration file following the example in `configs/preprocess_train_generate_example.yaml`.
Arguments in the training config:
- `preprocess`: Boolean. Creating a vocabulary or not.
- `train`: Boolean. Train a model or not.
- `generate`: Boolean. Inference , so generate with a trained model or not.
- `experiment_id`: integer. The ID of the current experiment.
- `features_requested`: list of str. The list of features we want to control the output for.
- `language`: `en` or `de`
- `arch`: currently only the Transformer is implemented
- `optimizer`: currently only Adam is supported
- `batch_size`: integer. Batch size during training.
- `update_freq`: integer. Number of batches of which the gradients get accumulated before updating the parameters. If set to 1 then the model is updated after each batch, if set to a larger value this increases the effective batchsize to `batch_size` * `update_freq`.
- `test_batch_size`: integer. Batch size during testing.
- `max_epochs`: integer. Maximum number of epochs to train the model.
- `patience`: integer. If validation performance does not improve for `patience` successive validations then early stopping is executed. Set to -1 if no early stopping should be happen.
- `beam_size`: integer. Beam size for inference at the testing step.
- `lr`: float. learning rate.

Run the script with `python fairseq_preprocess_train_generate.py --config configs/preprocess_train_generate_example.yaml`.<br>
Note that the logging information for the training will not be redirected to a file but printed to the command line. To store the information in a file add `> log_file.txt` at the end of the command to run the training script.

See the [Wiki](https://github.com/coli-saar/rewrite_text/wiki/Optional-Scripts-Tuning) for optional helpers for hyperparameter tuning. 

Note that the original ACCESS code uses early stopping using the SARI metrics, but this implementation doesn't.

## Inference: using a trained model for decoding
In the step above, the generation/inference uses the gold control token values. In contrast, this step allows for a free choice of control token values.

The control token values are given as arguments to the script `generate.py`.
The source file should contain one sentence per line. Before running the script, make sure to name the source file `test.txt`.

Required arguments of the script:
- `experiment-id`: Experiment ID of the trained model, i.e. the name of the folder in`./experiments` containing the `checkpoint` folder with the trained checkpoint (`checkpoint_best.pt`) and the vocabulary files created during training
- `data-dir`: Directory containing the src test.txt file.
- `language`: the language of the src file and the model, either `en` or `de`
- `beam`: beam size

Optional arguments - control token values:
- `dependency`
- `frequency`
- `length`
- `levenshtein`

An example for running this script: 

`python generate.py --experiment-id 5 --data-dir data_generetaion --language en --beam 5 --length 0.95 --frequency 0.75 --levenshtein 0.75`.


## Additional Scripts
The folder additional_scripts contains scripts that are not necessary for training and using the simplification model but that might be helpful for inspecting the data, plotting, etc. 

The [Wiki](https://github.com/coli-saar/rewrite_text/wiki#preprocessing) provides further information about the individual scripts and their usage. 
