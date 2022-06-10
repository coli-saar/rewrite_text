from contextlib import contextmanager, AbstractContextManager
import sys
import os
from collections import defaultdict
import re
from pathlib import Path
from itertools import zip_longest
import spacy
import yaml
import matplotlib.pyplot as plt
from utils.paths import get_data_preprocessed_dir
# TODO: need to add sentencepiece to the requirements then
import sentencepiece as spm
from .paths import data_auxiliary_dir, REPO_DIR

feature2spec_token = {"dependency": "MaxDep", "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven"}

def print_something():
    print("something")


@contextmanager
def open_files(filepaths, mode='r'):
    # pass a list of filenames as arguments and yield a list of open file objects
    # this function is useful also for readily preprocessed files that have text features
    files = []
    try:
        files = [Path(filepath).open(mode, encoding="utf-8") for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    # read in files (meant for 2: source and target) in parallel line by line and yield tuples of parallel strings
    # this function is useful also for readily preprocessed files that have text features
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [line.strip() if line is not None else None for line in parallel_lines]
            yield parallel_lines


# f1, f2 = "a.complex", "a.simple"
#
# for x in yield_lines_in_parallel([f1, f2], strict=True):
#     # print(len(x))
#     # for y in x:
#     #     print(y)
#     a, b = x
#     print("complex:", a)
#     print("simple:",b)

# for a, b in yield_lines_in_parallel([f1, f2], strict=True):
#     print(a)
#     print(b)

# TODO write code for writing into files (preprocessed and equipped with features)


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def format_control_features(feature_value, feature_token, source_string):
    pass


def prepend_feature_to_string(original_source_string, original_target_string,
                              feature_list, feature_value_dict, lang, path_to_tokenizer):
    # tokenize the original source and target sentence - for now with the spacy tokenizer
    # TODO: train a sentencepiece tokenizer - before running preprocess.py
    # include the special tokens as well (as user-defined symbols)
    # https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
    # Multiple files can be used to train it https://github.com/google/sentencepiece/issues/489
    # Use the train and val SRC and TGT

    # mapping: feature: special_token
    feature2spec_token = {"dependency": "MaxDep", "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven"}

    # define the spacy model
    lang_model1 = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if lang.lower() not in lang_model1:
        print("Language choice not supported, defaulting to English (other option: German)")
        lang = "en"

    nlp_model1 = spacy.load(lang_model1[lang])
    source_tokens = [t.text for t in nlp_model1(original_source_string) if t.text not in {" ", "  "}]
    to_be_prepended = ""
    for f in feature_list:
        v = str(round(feature_value_dict[f], 2))
        # print(f, v)
        prep = "<" + feature2spec_token[f] + "_" + v + "> "
        to_be_prepended += prep

    new_source = to_be_prepended + " ".join(source_tokens)
    new_target = " ".join([z.text for z in nlp_model1(original_target_string)])
    with open(REPO_DIR / "test_tokenization_spacy.txt", "a", encoding="utf-8") as tf:
        tf.write(new_source)
        tf.write("\n")
        tf.write(new_target)
        tf.write("\n\n")

    # TODO: check whether the replacement works as it should
    # TODO: decide whether to add English back and in which way, i.e. get English ccnet spm model?
    model_dir = data_auxiliary_dir / lang
    lang_model = {"de": model_dir / "de.sp.model"}
    if lang.lower() not in lang_model:
        print("Language choice not supported, defaulting to English (other option: German)")
        lang = "en"

    nlp_model = spm.SentencePieceProcessor()
    nlp_model.load(str(lang_model[lang]))

    source_tokens = nlp_model.encode_as_pieces(original_source_string)

    to_be_prepended = ""
    for f in feature_list:
        v = str(round(feature_value_dict[f], 2))
        #print(f, v)
        prep = "<" + feature2spec_token[f] + "_" + v + "> "
        to_be_prepended += prep

    new_source = to_be_prepended + " ".join(source_tokens)
    new_target = " ".join(nlp_model.encode_as_pieces(original_target_string))

    #print("New source: ", new_source, "\n", "New target: ", new_target, "-"*10)
    return new_source, new_target


def plot_histogram(x, feature_name, lang):
    n_bins = 80  # 50 in the binned version for NLG
    if feature_name in {"levenshtein"}:
        n_bins = 50  # 30 in the binned version for NLG
    plt.hist(x, density=False, bins=n_bins)  # density=False would make counts
    plt.ylabel('Count')
    plt.xlabel(feature_name)
    plt.title("Histogram for " + feature_name + " in " + lang)
    out_name = str(get_data_preprocessed_dir(lang) / feature_name) + ".png"
    plt.savefig(out_name)
    plt.clf()


def write_output_into_file(filename):
    pass


def match_dir_with_features(list_allowed_features, list_requested_features):
    assert set(list_requested_features).issubset(set(list_allowed_features))
    d = "_".join(sorted(list_requested_features))  # sort alphabetically and join into a string
    return d


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    """ Context manager to write both to stdout and to a file """
    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


def yield_lines(filepath, n_lines=float('inf'), prop=1):
    # if prop < 1:
    #     assert n_lines == float('inf')
    #     n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')


def parse_model_hypotheses(filepath):
    hypotheses_dict = defaultdict(list)
    for line in yield_lines(filepath):
        match = re.match(r'^H-(\d+)\t-?\d+\.\d+\t(.*)$', line)
        if match:
            sample_id, hypothesis = match.groups()
            hypotheses_dict[int(sample_id)].append(hypothesis)
    # Sort in original order
    return [hypotheses_dict[i] for i in range(len(hypotheses_dict))]


