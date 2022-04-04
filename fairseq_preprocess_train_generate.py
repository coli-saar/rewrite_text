"""
Steps: read in config in yaml with arguments for data directory, source/target "languages" (extensions),
model architecture and hyperparameters (apart from a Transformer model, give the option for an LSTM)



"""
import yaml
from pathlib import Path
import argparse
import os
import subprocess
from utils.helpers import match_dir_with_features, load_yaml
from with_fairseq.fairseq_base import preprocess_with_fairseq, train_with_fairseq, generate_with_fairseq
from utils.paths import get_data_preprocessed_dir, get_experiment_dir, check_if_dir_exists_and_is_empty, get_evaluation_dir
from with_fairseq.fairseq_base import preprocess_with_fairseq, train_with_fairseq, generate_with_fairseq, evaluation_automatic_metrics
from evaluation.feature_match_evaluate import parse_file_pair_return_analysis
import torch

#import torch.distributed as dist
#dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1) 

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="yaml config file preprocess/train/generate")
args = vars(parser.parse_args())

config = load_yaml(args["config"])
FEATURES_REQUESTED = sorted(config["features_requested"])
LANG = config["language"].lower()
EXP_ID = str(config["experiment_id"]).lower()
PREPROCESS, TRAIN, GENERATE = config["preprocess"], config["train"], config["generate"]
# LANG = "en"
# FEATURES_REQUESTED = ["dependency", "frequency", "length"]
# HYPERPARAMETERS
hyper = {"lr": float(config["lr"]), "batch_size": int(config["batch_size"]),
         "test_batch_size": int(config["test_batch_size"]), "beam_size": int(config["beam_size"])}

lang_allowed = {"en": "English", "de": "German"}
features_allowed = {"dependency", "frequency", "length", "levenshtein"}
splits_allowed = {"train", "valid", "test"}

# some checks
assert LANG in lang_allowed
assert set(FEATURES_REQUESTED).issubset(features_allowed)

# prepare names
# get the feature dir
input_to_preprocessing_suffix = match_dir_with_features(list(features_allowed), FEATURES_REQUESTED)
# get the entire path
dir_input_to_preprocessing = get_data_preprocessed_dir(LANG) / input_to_preprocessing_suffix
# /home/skrjanec/rewrite_text/data_preprocessed/en/dependency_frequency_length


destination_dir_fairseq_preprocessing = dir_input_to_preprocessing / "fairseq"

# if preprocess
if PREPROCESS:
    print("... STEP: PREPROCESSING")
    # the destination directory should be empty otherwise the code will raise an error and finish
    # here check if the destination exists (else create), and if it's empty
    check_if_dir_exists_and_is_empty(destination_dir_fairseq_preprocessing)

    preprocess_with_fairseq(data_directory=dir_input_to_preprocessing,
                            destination_directory=destination_dir_fairseq_preprocessing)



# if train
if TRAIN:
    print("... STEP: TRAINING")
    experiment_dir_full = get_experiment_dir(EXP_ID)
    checkpoint_suffix = "checkpoints"
    if not os.path.exists(experiment_dir_full):
        os.makedirs(experiment_dir_full)  # /home/skrjanec/rewrite_text/experiments/03

    train_with_fairseq(dir_with_preprocessed_files=destination_dir_fairseq_preprocessing,
                       experiment_dir=experiment_dir_full, dir_checkpoints_suffix=checkpoint_suffix,
                       lr=hyper["lr"], batch_size=hyper["batch_size"])


# if generate
if GENERATE:
    print("... STEP: INFERENCE")
    checkpoint_suffix = "checkpoints"
    experiment_checkpoint_dir_full = get_experiment_dir(EXP_ID) / checkpoint_suffix
    generate_with_fairseq(dir_with_test_data_and_vocab=destination_dir_fairseq_preprocessing,
                          dir_with_model_test_data_and_vocab=experiment_checkpoint_dir_full)

    torch.cuda.empty_cache()  # will this help with the OOM?
    # evaluate with EASSE for BLEU, SARI, FKGL and BERTScore, tokenize with Moses.
    # Write the results into a file in the same directory as  checkpoint_best.pt and generation2.out
    print("... AUTOMATIC EVALUATION")
    test_src_path, test_system_path = \
        evaluation_automatic_metrics(dir_with_model_test_data_and_vocab=experiment_checkpoint_dir_full)

    # evaluate also for the match in feature values between requested and actually generated features
    # use the paths test_src_path and test_system_path

    # use another function to handle: calling feature extraction, bin preparation
    # match evaluation
    parse_file_pair_return_analysis(test_src_path, test_system_path, FEATURES_REQUESTED, LANG)
