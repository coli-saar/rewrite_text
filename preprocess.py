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

import yaml
import argparse
from utils.helpers import load_yaml, yield_lines_in_parallel, prepend_feature_to_string, plot_histogram
from utils.paths import get_input_filepaths_dict, get_out_filepaths_dict, get_phase_suffix_pairs
from utils.feature_extraction import feature_bins_bundle_sentence, feature_bundle_sentence
from utils.prepare_word_embeddings_frequency_ranks import load_ranks
from utils.feature_bin_preparation import create_bins

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="yaml config file for preprocessing src and tgt")
args = vars(parser.parse_args())

config = load_yaml(args["config"])
FEATURES_REQUESTED = sorted(config["features"])
LANG = config["lang"].lower()

lang_allowed = {"en": "English", "de": "German"}
features_allowed = {"dependency", "frequency", "length", "levenshtein"}
splits_allowed = {"train", "valid", "test"}

# some checks
assert LANG in lang_allowed
assert set(FEATURES_REQUESTED).issubset(features_allowed)
print("... Preprocessing %s corpora and extracting features: %s " % (lang_allowed[config["lang"]],
                                                                     ", ".join(FEATURES_REQUESTED)))

input_file_paths = get_input_filepaths_dict(LANG)
output_file_paths = get_out_filepaths_dict(LANG, FEATURES_REQUESTED)

parallel_pairs_list = get_phase_suffix_pairs()
feature_bins = create_bins()

frequency_ranks = load_ranks(LANG)


def phase_open_process_write(src_tuple, tgt_tuple, in_src_PATH, in_tgt_PATH):
    """ ("phase", "src"), ("phase", "tgt") """
    # open the out files
    new_source = open(output_file_paths[src_tuple], "w", encoding="utf-8")
    new_target = open(output_file_paths[tgt_tuple], "w", encoding="utf-8")

    feature_dict_vals = {feat: [] for feat in FEATURES_REQUESTED}

    for src_sent, tgt_sent in yield_lines_in_parallel([in_src_PATH, in_tgt_PATH], strict=True):
        f_vals_bin, f_vals_exact = feature_bins_bundle_sentence(src_sent, tgt_sent, LANG, FEATURES_REQUESTED,
                                                                feature_bins, frequency_ranks)
        sent_src_new, sent_tgt_new = prepend_feature_to_string(src_sent, tgt_sent, FEATURES_REQUESTED,
                                                               f_vals_bin, LANG, "path_to_tokenizer")
        new_source.write(sent_src_new + "\n")
        new_target.write(sent_tgt_new + "\n")

        if config["analyze_features"]:
            for f, v in f_vals_exact.items():
                feature_dict_vals[f].append(v)


    # close the out files
    new_target.close()
    new_source.close()

    return feature_dict_vals


all_phases_all_feature_values = {feat: [] for feat in FEATURES_REQUESTED}

for phase in parallel_pairs_list:
    # phase is "train", "val" or "test"
    # tuple_src is (phase, "src") and tuple_tgt is (phase, "tgt")
    tuple_src, tuple_tgt = phase[0], phase[1]
    input_src_path = input_file_paths[tuple_src]
    input_tgt_path = input_file_paths[tuple_tgt]
    feature_d_values = phase_open_process_write(tuple_src, tuple_tgt, input_src_path, input_tgt_path)
    for f, v in feature_d_values.items():
        """ f is a str, v is a list """
        all_phases_all_feature_values[f].extend(v)

if config["analyze_features"]:
    for f, _x in all_phases_all_feature_values.items():
        plot_histogram(_x, f, LANG)
