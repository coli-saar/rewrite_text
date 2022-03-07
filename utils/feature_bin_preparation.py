"""
Pool feature extractors into a single function and bin their values
"""

# prepare the bins in advance
# this will circumvent the problem of first loading all instances, calculating their features, binning them and
# then iterating over the instances again to assign feature bins and write into file

# BINS
import numpy as np


def create_bins():
    frequency_bins = np.arange(0.05, 2.5, 0.05)  # min 0.05, max 2.0
    dependency_bins = np.arange(0.05, 2.5, 0.05)
    length_bins = np.arange(0.05, 2.5, 0.05)
    levenshtein_bins = np.arange(0.05, 1.5, 0.05)  # min 0.05, max 1.0
    return {"frequency": frequency_bins, "dependency": dependency_bins, "length": length_bins,
             "levenshtein": levenshtein_bins}

