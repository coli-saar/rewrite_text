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
import sentencepiece as spm
from .paths import get_data_auxiliary_dir

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


def load_yaml(file_path):
    file_path = file_path.strip()
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_tokenizer(tokenizer_type, lang):

    if tokenizer_type not in {"spacy", "sentpiece"}:
        print(
            "Tokenizer choice not supported, defaulting to SentencePiece tokenizer (other option: spacy language model)"
        )
    if lang.lower() not in {"en", "de"}:
        print("Language choice not supported, defaulting to English (other option: German)")

    if tokenizer_type == "spacy":
        lang_models = {"de": "de_core_news_sm", "en": "en_core_web_sm"}
        tokenizer_model = spacy.load(lang_models[lang])

    else:
        model_dir = get_data_auxiliary_dir(lang)
        lang_models = {"de": model_dir / "de.sp.model", "en": model_dir / "en.sp.model"}
        tokenizer_model = spm.SentencePieceProcessor()
        tokenizer_model.load(str(lang_models[lang]))

    return tokenizer_model


def format_control_features(feature_value, feature_token, source_string):
    pass


def prepend_feature_to_string(original_source_string, original_target_string,
                              feature_list, feature_value_dict, tokenizer_type, tokenizer_model):
    # tokenize the original source and target sentence
    # include the special tokens as well (as user-defined symbols)
    # https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
    # Multiple files can be used to train it https://github.com/google/sentencepiece/issues/489
    # Use the train and val SRC and TGT

    # mapping: feature: special_token
    feature2spec_token = {"dependency": "MaxDep", "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven"}

    if tokenizer_type == "spacy":
        source_tokens = run_spacy_tokenizer(original_source_string, tokenizer_model)
        target_tokens = run_spacy_tokenizer(original_target_string, tokenizer_model)
    else:
        source_tokens = run_sentencepiece_tokenizer(original_source_string, tokenizer_model)
        target_tokens = run_sentencepiece_tokenizer(original_target_string, tokenizer_model)

    to_be_prepended = ""
    for f in feature_list:
        v = str(round(feature_value_dict[f], 2))
        prep = "<" + feature2spec_token[f] + "_" + v + "> "
        to_be_prepended += prep

    new_source = to_be_prepended + " ".join(source_tokens)
    new_target = " ".join(target_tokens)

    return new_source, new_target


def run_spacy_tokenizer(original_string, spacy_model):
    tokenized_string = [t.text for t in spacy_model(original_string) if t.text not in {" ", "  "}]
    return tokenized_string


def run_sentencepiece_tokenizer(original_string, sentpiece_model):
    tokenized_string = sentpiece_model.encode_as_pieces(original_string)
    return tokenized_string


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


