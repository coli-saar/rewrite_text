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

In order to use the 'sentencepiece' tokenizer (as in the original ACCESS code) you need to place a trained [SentencePiece tokenizer](https://github.com/google/sentencepiece) in the 'data_auxiliary/[en|de]' folder. We use the pretrained sentencepiece tokenizers from [here](https://github.com/facebookresearch/cc_net). 

## Preprocessing: feature extraction
The preprocessing involves several steps:
1) Prepare word frequency ranks. Run the function `write_ranks_into_file` from the script in `utils/prepare_word_embeddings_frequency_ranks.py`. This should create a json file in `data_auxiliary/en` (or `/de`) with `word:rank` pairs.
2) Prepare a configuration file for preprocessing following the example in `configs/preprocess_dummy.yaml`
Arguments in the preprocessing config:
   - `experiment_id`: ignore this
   - `lang`: the language of the dataset, either `en` or `de` supported for now
   - `features`: a list of features to be extracted from the dataset. The options are dependency, frequency, length, levenshtein. It's best to run the preprocessing with feature extraction once because it is time-consuming. The features that won't be used in the experiments can be removed from files after.
   - `analyze_features`: a functionality that plots histograms for the feature values. Boolean.
   - `overwrite`: ignore this
3) Prepare the directory where the newly processed data will be stored given the requested features, use `mkdir` to create e.g. `data_preprocessed/en/frequency_length` if you plan to preprocess English data and extract frequency and length features.
4) Run the preprocessing script, `python preprocess.py --config configs/[config_file].yaml --tokenizer [tokenizer_type]` where [tokenizer_type] can be either 'spacy' (for using the spacy lm for tokenization) or 'sentpiece' (for sentence piece tokenization). Depending on the dataset size and the features this step might take some time, so it's best to run it with `nohup` or `screen`.


Once finished, the preprocessed files should be located in a subdirectory of `data/en` or `data/de`. For example, if the features [dependency, frequency] were selected, the preprocessed files will be in `dependency_frequency/`.

## Training a sequence-to-sequence model
For training first prepare the configuration file following the example in `configs/preprocess_train_generate_wikilarge.yaml`.
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
- `test_batch_size`: integer. Batch size during testing.
- `beam_size`: integer. Beam size for inference at the testing step.
- `lr`: float. learning rate.


Run the script with `python fairseq_preprocess_train_generate.py --config configs/preprocess_train_generate_wikilarge.yaml`.

Note that the original ACCESS code uses early stopping using the SARI metrics, but this implementation doesn't.

## Inference: using a trained model for decoding
In the step above, the generation/inference uses the gold control token values. In contrast, this step allows for a free choice of control token values.

The control token values are given as arguments to the script `generate.py`.
Before running the script, make sure to put the source file into `data_generation` and name it `test.src-tgt.src`.

Required arguments of the script:
- `experiment_id`: ID of the experiment where th model was trained.
- `data_dir`: directory with the test.src-tgt.src file
- `language`: the language of the src file and the model, either `en` or `de`
- `beam`: beam size

Optional arguments - control token values:
- `dependency`
- `frequency`
- `length`
- `levenshtein`

An example for running this script: 

`python generate.py --experiment-id 5 --data-dir data_generation --language en --beam 5 --length 0.95 --frequency 0.75 --levenshtein 0.75`.
