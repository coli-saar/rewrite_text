from utils.paths import get_data_shard_dir, get_data_original_dir
from pathlib import Path
import argparse


def split_into_shards(file_name: str, lang: str, l_per_s: int, skip):
    """
    Runs the shard splitting for one file and creates the shard files
    :param file_name: name of the file to split
    :param lang: the language, 'de' or 'en'
    :param l_per_s: the number of lines, i.e. sentences, per shard
    :param skip: the number of lines to skip at the beginning of the corpus file
    """

    input_dir = get_data_original_dir(lang)
    output_dir = get_data_shard_dir(lang) / "original"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    phase = file_name.split(".")[0]
    suffix = file_name.split(".")[1]

    lines_current_shard = []
    n_lines_shard = 0
    shard_id = 0
    if skip:
        shard_id = 1

    with open(input_dir / file_name, "r", encoding="utf-8") as corpus:

        for n_line, line in enumerate(corpus):
            if n_line < skip:
                continue

            if n_lines_shard == l_per_s:
                shard_name = phase + "_" + str(shard_id) + "." + suffix
                write_to_shard(lines_current_shard, output_dir / shard_name)
                lines_current_shard = []
                n_lines_shard = 0
                shard_id += 1

            lines_current_shard.append(line)
            n_lines_shard += 1

        if lines_current_shard:
            shard_name = phase + "_" + str(shard_id) + "." + suffix
            write_to_shard(lines_current_shard, output_dir / shard_name)


def write_to_shard(line_list, shard_path):
    """
    writes the lines from line_list into the file at shard_path
    :param line_list: list(str), list of the sentences to write in the file
    :param shard_path: path/name of the shard file
    """
    with open(shard_path, "w", encoding="utf-8") as shard:
        for line in line_list:
            shard.write(line)


def split_phase_data(lang: str, l_per_s: int, phase='train', skip=0):
    """
    Splits the [phase].src and [phase].tgt files into several smaller files, each consisting of l_per_s sentences
    e.g. train_0.src and train_0.tgt, train_1.src and train_1.tgt, ... in the folder data_shards/[lang]/
    Order of the sentences is preserved in the shards
    :param lang: the language, 'de' or 'en'
    :param l_per_s: the number of lines, i.e. sentences, per shard, last shard may contain less sentences
    :param phase: the data set split that should be split into shards, default is to split the training split further
    :param skip: the number of lines to skip at the beginning of the corpus file, e.g. if the first 1000 lines should
                not be included in the shard because they were already preprocessed set skip=1000
                if provided then the enumeration of the shards starts at 0 instead of at 0
    """
    source_file = f'{phase}.src'
    target_file = f'{phase}.tgt'
    split_into_shards(source_file, lang, l_per_s, skip)
    split_into_shards(target_file, lang, l_per_s, skip)


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--lang", required=True, help="the language, i.e. the name subfolder of the data folder "
                                                          "where the corpus is")
    arg_parser.add_argument("--lines", required=True, help="the number of sentences each shard should consist of")
    arg_parser.add_argument("--split", required=False, help="the data set split to split into shards, default is 'train'")
    arg_parser.add_argument("--skip", required=False, help="number of lines at the beginning of the corpus file that"
                                                           "should be skipped")
    args = vars(arg_parser.parse_args())

    if args["skip"] and args["split"]:
        split_phase_data(lang=args["lang"], l_per_s=args["lines"], phase=args["split"], skip=args["skip"])
    elif args["skip"]:
        split_phase_data(lang=args["lang"], l_per_s=args["lines"], skip=args["skip"])
    elif args["split"]:
        split_phase_data(lang=args["lang"], l_per_s=args["lines"], phase=args["split"])
    else:
        split_phase_data(lang=args["lang"], l_per_s=args["lines"])

