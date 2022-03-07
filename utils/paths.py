import git
from pathlib import Path

repo = git.Repo('.', search_parent_directories=True)
print(repo.working_tree_dir)

# REPO_DIR = Path(__file__).resolve().parent.parent.parent
# EXP_DIR = REPO_DIR / 'data_auxiliary'

def get_output_filepath(list_of_features):
    pass


