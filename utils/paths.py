import git
from pathlib import Path
from itertools import combinations, product

repo = git.Repo('.', search_parent_directories=True)
REPO_DIR = repo.working_tree_dir


# REPO_DIR = Path(__file__).resolve().parent.parent.parent
# EXP_DIR = REPO_DIR / 'data_auxiliary'
# print(REPO_DIR)

data_original_dir = Path(REPO_DIR) / "data"
data_preprocessed_dir = Path(REPO_DIR) / "data_preprocessed"
data_auxiliary_dir = Path(REPO_DIR) / "data_auxiliary"
configs_dir = Path(REPO_DIR) / "configs"
experiments_dir = Path(REPO_DIR) / "experiments"

splits = ["train", "val", "test"]
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


def get_data_original_dir(lang):
    return data_original_dir / lang


def get_experiment_dir(exp_id):
    return experiments_dir / str(exp_id)


def get_data_preprocessed_dir(lang):
    return data_preprocessed_dir / lang


def get_configs_dir(exp_id):
    return configs_dir / exp_id


def get_data_auxiliary_dir(lang):
    return data_auxiliary_dir / lang


def get_data_filepath(_features, phase, suffix, lang):
    feats = "_".join(_features)
    filename = f'{phase}.{suffix}'
    return get_data_preprocessed_dir(lang) / feats / filename


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

#di = get_out_filepaths_dict("de", ["dependency", "frequency", "length"])
# ('train', 'src'): PosixPath('/home/skrjanec/rewrite_text/data_preprocessed/de/dependency_frequency_length/train.src')
#import pdb; pdb.set_trace()

# din = get_input_filepaths_dict("de")
# for k, v in din.items():
#     print(k, v)
#import pdb; pdb.set_trace()

# print(sorted(din.keys()))

