"""
Using the preprocessing.py code, I equipped the source (complex) side of the WikiLarge corpus with 4 relative features:
- Dependency
- Word Frequency
- Length
- Levenshtein similarity

However, Martin et al. (2020) found that using all features except for Dependency gives the best performance in terms
of SARI and FKGL. I want to create versions of the source side of WikiLarge with different feature combinations:
- all 4
- only 3
- only 2
- only 1

"""
import git
from pathlib import Path
import os
from itertools import combinations
from utils.helpers import yield_lines
from evaluation.feature_match_evaluate import remove_features_from_str

LANG = "en"
splits_allowed = ["train", "valid", "test"]
features_allowed = ["dependency", "frequency", "length", "levenshtein"]
feature2spec_token = {"dependency": "MaxDep", "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven"}
suffix_allowed = ".src"

repo = git.Repo('.', search_parent_directories=True)
REPO_DIR = repo.working_tree_dir
path_to_4features_dir = Path(REPO_DIR) / "data_preprocessed" / LANG / "dependency_frequency_length_levenshtein"


def create_f_combinations(list_of_features):
    """ Given the list of features, return a list of features where every element is a list of features of length
    between 1 and 3; 4 is already covered. Order by descending length
    """
    list_combinations = []
    for n in range(1, len(list_of_features)):
        list_combinations += list(combinations(list_of_features, n))

    list_combinations = [list(x) for x in list_combinations]
    list_combinations.reverse()
    return list_combinations


def make_path_make_dir(f_combination):
    # for a given list of requested features, create a full path to this directory and create a dir
    part1 = Path(REPO_DIR) / "data_preprocessed" / LANG
    # concatenate the feature names
    part2 = "_".join(f_combination)
    datadir = part1 / part2
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    print("Created directory ", str(datadir))
    return datadir


def select_relevant_features_return_str(f_combination, feat_dict):
    """
    :param f_combination: a list with desired features, e.g. ['dependency', 'levenshtein']
    :param feat_dict: a dict with feature tokens and their values for the current sentence, e.g.
        "MaxDep": 1.0, "FreqRank": 0.8, "Length": 1.05, "Leven": 0.45
    :return: a str that is to be prepended to the src sentence, e.g. "<MaxDep_1.0> <Leven_0.45> "
    """
    f_relevant_d = {feature2spec_token[f_name]: feat_dict[feature2spec_token[f_name]] for f_name in f_combination}
    s = ""
    for k, v in f_relevant_d.items():
        s += "<" + k + "_" + v + "> "
    return s


all_combinations = create_f_combinations(features_allowed)
for combi in all_combinations:
    c_datadir = make_path_make_dir(combi)
    for split in splits_allowed:
        # open a file
        c_suffix = split + suffix_allowed
        new_file_path = c_datadir / c_suffix
        new_file = open(new_file_path, "w")
        src_file_path = path_to_4features_dir / c_suffix
        for f_sentence in yield_lines(src_file_path):
            f_dict, sentence = remove_features_from_str(f_sentence)
            prepend = select_relevant_features_return_str(combi, f_dict)
            sentence = prepend + sentence
            new_file.write(sentence + "\n")
        new_file.close()
        print("... Wrote file in ", str(new_file_path))

# create feature combinations, put them into a list

# for every feature combination, do
#   create a directory in data_preprocessed/en/
#   for every split in splits_allowed:
#       open the split in path_to_4features_dir / split.src:
#           yield_lines
#           pass the line to remove_features_from_str; feat_dict, sentence = remove_features_from_str(line)
#           from the feat_dict, keep only those from feature_combination
#               prepend again to the sentence and write into open file
#           close the open file


