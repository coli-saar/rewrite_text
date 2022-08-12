"""
Read in command line arguments from argparse

folder of the model:
folder of the test src file
# src-file: full path to the source file
# model-path: full path to the .pt model
beam: the size of the beam
dependency: feature value float
frequency: feature value float
length: feature value float
lev: feature value float
"""

import argparse
from pathlib import Path
import os
import sys
from fairseq import options
from fairseq_cli import generate
from generation.helpers_generation import update_requested_features_with_bins, copy_vocab_files, preprocess_src_file, \
    prepare_special_token_string
from utils.paths import get_experiment_dir, get_repo_dir
from utils.helpers import parse_model_hypotheses, log_stdout

parser = argparse.ArgumentParser()
# parser.add_argument("--model-dir", required=True, help="dir with the checkpoint_best.pt model")
parser.add_argument("--experiment-id", required=True, help="ID of the experiment, checkpoint_best.pt will be used")
parser.add_argument("--data-dir", required=True, help="dir with the test.txt file to rewrite")
parser.add_argument("--language", required=True, help="the language of input text, options: en, de")
parser.add_argument("--beam", required=True, help="beam size, default 8", default=8, type=int)
parser.add_argument("--dependency", required=False, help="maximum dependency tree depth ratio, from 0.05 to 2.45")
parser.add_argument("--frequency", required=False, help="frequency rank ratio, from 0.05 to 2.45, recommended 0.95")
parser.add_argument("--length", required=False, help="length ratio, from 0.05 to 2.45, recommended 0.75")
parser.add_argument("--levenshtein", required=False, help="Levenshtein ratio, from 0.05 to 1.0, recommended 0.75")

args = vars(parser.parse_args())

feature2spec_token = {"dependency": "MaxDep", "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven"}

features_values = {}
if args["dependency"]:
    features_values["dependency"] = float(args["dependency"])

if args["frequency"]:
    features_values["frequency"] = float(args["frequency"])

if args["length"]:
    features_values["length"] = float(args["length"])

if args["levenshtein"]:
    features_values["levenshtein"] = float(args["levenshtein"])

# update the feature_values dict with bin values, round them to 2 decimals too
features_values = update_requested_features_with_bins(features_values)

text = " \n".join([f_name + ": " + str(f_val) for f_name, f_val in features_values.items()])
print("Generating with features: \n" + text)

# preparing paths:
model_dir = get_experiment_dir(args["experiment_id"]) / "checkpoints"
model_path = model_dir / "checkpoint_best.pt"
if not os.path.exists(model_dir):
    sys.exit("Error: This model directory does not exist" + str(model_dir))

data_dir = Path(get_repo_dir()) / args["data_dir"]
src_file_path = data_dir / "test.txt"

suffix = "test.src-tgt.src"
#prep_src_file_name = args["source-file"].split('.')[:-1]    # remove file ending from input source file
#prep_src_file_name = '.'.join(prep_src_file_name) + suffix  # and add new suffix .src-tgt.src to file name
src_file_destin = data_dir / suffix
if not os.path.exists(data_dir):
    sys.exit("Error: This data directory does not exist" + str(data_dir))

source_lang = "src"
target_lang = "tgt"

# copy the vocabulary dict.* files from the model_dir into the same dir as the test src file
copy_vocab_files(origin_dir=model_dir, destination_dir=data_dir)

# PREPROCESSING #
# read in the src-file
# to each line prepend the feature values
# save the file into experiment_id_test.src-tgt.src
special_token_str = prepare_special_token_string(features_values, feature2spec_token)
preprocess_src_file(src_file_path, src_file_destin, special_token_str, args['language'])


# GENERATING: INFERENCE #
print("+++ INFERENCE +++")
def generate_main():
    # fairseq.generate --> write the output into a temp file
    inference_args = [data_dir, "--path", model_path, "--batch-size", 12, "--beam", args["beam"],
                      "--dataset-impl", "raw", "--source-lang", source_lang, "--target-lang", target_lang]
    print("ARGUMENTS FOR DECODING", inference_args)
    inference_args = [str(a) for a in inference_args]
    inference_parser = options.get_generation_parser()
    inf_args = options.parse_args_and_arch(inference_parser, inference_args)

    out_temp = str(data_dir / "generation.out")
    with log_stdout(out_temp, mute_stdout=True):
        generate.main(inf_args)
    # parse this file to fetch out the hypotheses H-N, order them 0, 1, 2...
    ordered_hypotheses = parse_model_hypotheses(out_temp)
    suffix_out = args["experiment_id"] + "_generation.out"
    out_file2 = str(data_dir / suffix_out)
    with open(out_file2, "w") as fout:
        for line in ordered_hypotheses:
            fout.write(line[0] + "\n")


generate_main()



