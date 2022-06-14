import json
from utils.paths import get_data_auxiliary_dir


def read_in_embeddings_return_frequency_rank_dict(path_to_embeddings):
    """
    :param path_to_embeddings: path to the file with word embeddings: one word per line, sorted by descending frequency
    :return: a dictionary, word as key, frequency rank (an integer) as value. More frequent words have a higher rank
    (lower int!)
    """
    ranks = dict()
    first, last = "", ""
    j = 0
    with open(path_to_embeddings, "r", encoding="utf-8") as f:
        for i, word_line in enumerate(f):
            # the first line is not a word embedding
            if i == 0:
                continue
            if word_line:
                word = word_line.split()[0].strip()
                ranks[word] = i
                if i == 1:
                    first = word
                if i > j:
                    last = word
                    j = i

    print("Vocabulary size %d, most frequent token %s, least frequent token %s" % (len(ranks), first, last))
    return ranks


def write_ranks_into_file(lang):
    if lang.lower() not in {"en", "de"}:
        print("Language choice not supporting, defaulting to English")
    if lang == "de":
        embedding_path = get_data_auxiliary_dir(lang) / "cc.de.300.vec"
    else:
        embedding_path = get_data_auxiliary_dir(lang) / "cc.en.300.vec"
    ranks = read_in_embeddings_return_frequency_rank_dict(embedding_path)
    with open(get_data_auxiliary_dir(lang) / 'frequency_ranks.json', "w") as fout:
        json.dump(ranks, fout)


def load_ranks(lang):
    if lang.lower() not in {"en", "de"}:
        print("Language choice not supporting, defaulting to English")

    ranking_file_path = get_data_auxiliary_dir(lang) / "frequency_ranks.json"

    with open(ranking_file_path, "r", encoding="utf-8") as fin:
        ranks = json.load(fin)
    return ranks


#write_ranks_into_file()
if __name__=="__main__":
    write_ranks_into_file("de")