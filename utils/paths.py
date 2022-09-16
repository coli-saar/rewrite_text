import os
from pathlib import Path
from itertools import combinations, product

REPO_DIR = Path(__file__).resolve().parent.parent
# EXP_DIR = REPO_DIR / 'data_auxiliary'
# print(REPO_DIR)

data_original_dir = Path(REPO_DIR) / "data"
data_preprocessed_dir = Path(REPO_DIR) / "data_preprocessed"
data_auxiliary_dir = Path(REPO_DIR) / "data_auxiliary"
data_shard_dir = Path(REPO_DIR) / "data_shards"
configs_dir = Path(REPO_DIR) / "configs"
experiments_dir = Path(REPO_DIR) / "experiments"
additional_script_dir = Path(REPO_DIR) / "additional_scripts"

evaluation_dir = Path(REPO_DIR) / "evaluation"

splits = ["train", "valid", "test"]
suffixes = ["src", "tgt"]
features = ["dependency", "frequency", "length", "leven"]


def create_feature_combinations():
    """ Return a list of sorted lists"""
    combos = []
    for i in range(len(features)+1):
        for _tuple in combinations(features, i):
            if _tuple:
                combos.append(sorted(list(_tuple)))
    return combos


def get_repo_dir():
    return REPO_DIR


def get_data_original_dir(lang):
    return data_original_dir / lang


def get_experiment_dir(exp_id):
    return experiments_dir / str(exp_id)


def get_data_preprocessed_dir(lang):
    dir_path = data_preprocessed_dir / lang
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def get_configs_dir(exp_id):
    return configs_dir / exp_id


def get_data_auxiliary_dir(lang):
    return data_auxiliary_dir / lang


def get_data_shard_dir(lang):
    Path(data_shard_dir).mkdir(parents=True, exist_ok=True)
    lang_path = data_shard_dir / lang
    Path(lang_path).mkdir(parents=True, exist_ok=True)
    return lang_path


def get_evaluation_dir():
    return evaluation_dir


def get_data_filepath(_features, phase, suffix, lang):
    feats = "_".join(_features)
    filename = f'{phase}.{suffix}'
    dir_path = get_data_preprocessed_dir(lang) / feats
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path / filename


def get_data_filepath_original(phase, suffix, lang):
    filename = f'{phase}.{suffix}'
    return get_data_original_dir(lang) / filename


# = create_feature_combinations()


def get_out_filepaths_dict(lang, requested_features):
    return {(phase, suffix): get_data_filepath(requested_features, phase, suffix, lang)
            for phase, suffix in product(splits, suffixes)}


def get_input_filepaths_dict(lang):
    return {(phase, suffix): get_data_filepath_original(phase, suffix, lang)
            for phase, suffix in product(splits, suffixes)}


def get_phase_suffix_pairs():
    file_pairs = []
    for phase in splits:
        phase_list = []
        for suffix in suffixes:
            phase_list.append((phase, suffix))
        file_pairs.append(sorted(phase_list))
    return file_pairs  # [ [("train", "src), ("train", "tgt")], ... ]


# functions to get the folders and paths for the splitted corpus

def get_out_shardpath_dict(shard_name, lang, requested_features):
    path_dict = dict()
    parent_dir_path = get_data_shard_dir(lang) / "preprocessed"
    feats = "_".join(requested_features)
    dir_path = parent_dir_path / feats
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    for suffix in suffixes:
        file_name = f'{shard_name}.{suffix}'
        path_dict[(shard_name, suffix)] = dir_path / file_name

    return path_dict


def get_input_shardpath_dict(shard_name, lang):
    path_dict = dict()
    dir_path = get_data_shard_dir(lang) / "original"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        file_name = f'{shard_name}.{suffix}'
        path_dict[(shard_name, suffix)] = dir_path / file_name
    return path_dict



def check_if_dir_exists_and_is_empty(dir_path):
    # if the path to dir exists, delete its contents
    if os.path.exists(dir_path):
        [f.unlink() for f in Path(dir_path).glob("*") if f.is_file()]
    # if the dir doesn't exist create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


#di = get_out_filepaths_dict("de", ["dependency", "frequency", "length"])
# ('train', 'src'): PosixPath('/home/skrjanec/rewrite_text/data_preprocessed/de/dependency_frequency_length/train.src')
#import pdb; pdb.set_trace()

# din = get_input_filepaths_dict("de")
# for k, v in din.items():
#     print(k, v)
#import pdb; pdb.set_trace()

# print(sorted(din.keys()))
