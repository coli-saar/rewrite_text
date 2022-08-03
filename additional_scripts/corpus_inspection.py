import os
from collections import defaultdict


def search_zb(data_dir):
    """
    Counts how often a sentence was erroneously split into 2 sentences at z.B.
    i.e. into [beginning sentence z.] and [B. rest of sentence]
    Prints the counts for the train, valid and test split as well as the total count to the command line
    Note: occurrences of z. and B. are counted separately -> counts are likely too high
    """
    counts = defaultdict(int)
    counts["test.src"] = 0
    counts["test.tgt"] = 0
    counts["train.src"] = 0
    counts["train.tgt"] = 0
    counts["valid.src"] = 0
    counts["valid.tgt"] = 0

    for file in os.listdir(data_dir):

        with open(data_dir + "/" + file, "r", encoding="utf-8") as corpus:
            for line in corpus:
                sentence = line.strip()

                if sentence[:2] == "B.":
                    counts[file] += 1
                if sentence[-2:] == "z.":
                    counts[file] += 1

    counts["total"] = 0
    for c in counts.values():
        counts["total"] += c

    for key, value in counts.items():
        print(f'{key}: {value}')


def search_ca(data_dir):
    """
    Counts how often a sentence ends with 'ca.' in the individual data splits as well as in the total corpus
    i.e. how often a sentence was erroneously split into a new sentence after ca.
    """
    counts = defaultdict(int)
    counts["test.src"] = 0
    counts["test.tgt"] = 0
    counts["train.src"] = 0
    counts["train.tgt"] = 0
    counts["valid.src"] = 0
    counts["valid.tgt"] = 0

    for file in os.listdir(data_dir):

        with open(data_dir + "/" + file, "r", encoding="utf-8") as corpus:
            for line in corpus:
                sentence = line.strip()

                if sentence[-3:] == "ca.":
                    counts[file] += 1

    counts["total"] = 0
    for c in counts.values():
        counts["total"] += c

    for key, value in counts.items():
        print(f'{key}: {value}')


if __name__=="__main__":
    search_ca("../data/de")
