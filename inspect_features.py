from collections import defaultdict, Counter
from utils.paths import get_data_preprocessed_dir
from utils.helpers import feature2spec_token
import matplotlib.pyplot as plt
import argparse
import numpy as np


def extract_features(file_path: str):
    """
    :param file_path: path to the preprocessed source file, i.e. the file that includes the features that were extracted
                        during the preprocessing
    :return: a dictionary with the feature names as keys and a list of all feature value occurrences
    """

    feature_dict = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as prep_corp:
        for line in prep_corp:
            splitted_line = line.strip().split()

            # features are always prepended to the sentences
            # -> take as many features as were extracted during preprocessing
            for el in splitted_line:
                if el[0] == "<":
                    # remove < and > from beginning and end
                    feature_name_val = el[1:-1].split("_")
                    feature_name = feature_name_val[0]
                    feature_val = float(feature_name_val[1])
                    feature_dict[feature_name].append(feature_val)
                else:
                    break
    return feature_dict


def make_plot(x, feature_name, lang, plot_dir, split):
    """
    Creates the actual histogram plot
    Bins for the histogram are the same as the feature bins created during preprocessing
    :param x: the list of all features values
    :param feature_name: the name of the feature that gets plotted
    :param lang: language, 'de' or 'en'
    :param plot_dir: directory where plots get saved
    :param split: type of the split, 'train', 'valid' or 'test'
    :return None
    """
    # make the same bins as for the feature values
    n_bins = np.arange(0.01, 2.05, 0.05)
    # if going from 0.05 - 0.1 and 0.1 - 0.15 always two bins get merged for some reason
    # therefore start at 0.01
    if feature_name in {"levenshtein"}:
        n_bins = np.arange(0.01, 1.05, 0.05)

    plt.hist(x, density=False, bins=n_bins, rwidth=0.9)
    plt.ylabel('Count')
    plt.xlabel(feature_name)
    plt.title("Histogram for " + feature_name + " in " + lang + " " + split)
    file_name = feature_name + "_" + split + ".png"
    out_name = plot_dir + "/" + file_name
    plt.savefig(out_name)
    plt.clf()


def plot_features_split(features, feature_dict, lang, plot_dir, split):
    """
    :param features: list of the feature names
    :param feature_dict: dictionary with the feature names as keys and list of feature values as values
    :param lang: language, 'de' or 'en'
    :param plot_dir: directory where plots get saved
    :param split: type of the split, 'train', 'valid' or 'test'
    :return None
    """

    for feat_name in features:
        feature_key = feature2spec_token[feat_name]
        feature_values = feature_dict[feature_key]
        make_plot(feature_values, feat_name, lang, plot_dir, split)


def plot_features_corpus(lang, features, plot_dir):
    """
    Function to plot the frequency of the values of all features with all values
    Creates one plot per feature and split (i.e. training, validation and test split)
    Additionally creates one plot for each feature for the complete corpus

    :param lang: language
    :param features: list of the feature names that should be plotted (needs to be the same as the features used
                    during preprocessing in order to access the data right folder)
    :param plot_dir: directory where the plots should be saved
    return creates the plots and prints the counts for each feature values for the complete corpus
    """
    preprocessed_data_dir = get_data_preprocessed_dir(lang)
    features.sort()
    feature_folder = "_".join(features)

    # read features from preprocessed files
    train_features = extract_features(preprocessed_data_dir / feature_folder / "train.src")
    val_features = extract_features(preprocessed_data_dir / feature_folder / "valid.src")
    test_features = extract_features(preprocessed_data_dir / feature_folder / "test.src")

    # make the historgrams for the individual splits
    plot_features_split(features, train_features, lang, plot_dir, "train_split")
    plot_features_split(features, val_features, lang, plot_dir, "val_split")
    plot_features_split(features, test_features, lang, plot_dir, "test_split")

    # make the histogram for the complete corpus
    total_features = train_features.copy()
    for key, values in val_features.items():
        total_features[key].extend(values)
    for key, values in test_features.items():
        total_features[key].extend(values)
    plot_features_split(features, total_features, lang, plot_dir, "complete_corpus")

    for feats, vals in total_features.items():
        print(feats)
        c = Counter(vals)
        print(c)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    parser.add_argument("--out")
    args = vars(parser.parse_args())

    features = ["dependency", "frequency", "length", "levenshtein"]

    plot_features_corpus(args["lang"], features, args["out"])

