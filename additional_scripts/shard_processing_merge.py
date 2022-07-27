from utils.paths import get_data_shard_dir, get_data_filepath
from utils.helpers import load_tokenizer, load_yaml
import argparse


def merge_preprocessed_shards(phase: str, lang: str, shards: list, features: list):
    """
    :param phase: the data set split that the shards to put together belong to, 'train', 'test' or 'valid'
    :param lang: language
    :param shards: list of the ids of the shards that should be merged together in the same order as in the list
    :param features: list of the features that were extracted
    """

    preprocessed_shard_dir = get_data_shard_dir(lang) / "preprocessed" / "_".join(features)

    for suffix in ["src", "tgt"]:
        preprocessed_data = get_data_filepath(features, phase, suffix, lang)
        with open(preprocessed_data, "w", encoding="utf-8") as out_file:

            # for the test and valid data only move the preprocessed data to the target directory
            if len(shards) == 0:
                shard_name = phase + "." + suffix
                with open(preprocessed_shard_dir / shard_name, "r", encoding="utf-8") as shard:
                    for line in shard:
                        out_file.write(line)

            # for the train data add all preprocessed files into one file
            else:
                for shard_id in shards:
                    shard_name = phase + "_" + str(shard_id) + "." + suffix
                    with open(preprocessed_shard_dir / shard_name, "r", encoding="utf-8") as shard:
                        for line in shard:
                            # skip empty lines and end of shard files
                            if line == "\n":
                                continue
                            out_file.write(line)


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", required=True, help="configuration file for merging the shards")
    args = vars(arg_parser.parse_args())

    config = load_yaml(args["config"])
    FEATURES_REQUESTED = sorted(config["features"])
    LANG = config["lang"].lower()
    PHASES = config["phases"]

    for phase in PHASES:
        phase_key = phase + "_ids"
        FILE_IDS = config[phase_key]
        merge_preprocessed_shards(phase, LANG, FILE_IDS, FEATURES_REQUESTED)


