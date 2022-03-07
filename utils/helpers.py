from contextlib import contextmanager, AbstractContextManager
from pathlib import Path
from itertools import zip_longest


def print_something():
    print("something")


@contextmanager
def open_files(filepaths, mode='r'):
    # pass a list of filenames as arguments and yield a list of open file objects
    # this function is useful also for readily preprocessed files that have text features
    files = []
    try:
        files = [Path(filepath).open(mode) for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    # read in files (meant for 2: source and target) in parallel line by line and yield tuples of parallel strings
    # this function is useful also for readily preprocessed files that have text features
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [line.strip() if line is not None else None for line in parallel_lines]
            yield parallel_lines


f1, f2 = "a.complex", "a.simple"

for x in yield_lines_in_parallel([f1, f2], strict=True):
    print(len(x))
    for y in x:
        print(y)

# TODO write code for writing into files (preprocessed and equipped with features)


def write_output_into_file(filename):
    pass
