import shutil
import os
from pathlib import Path
import contextlib
import subprocess
from fairseq import options
from fairseq_cli import preprocess, train, generate
from utils.paths import get_data_preprocessed_dir, get_evaluation_dir
from utils.helpers import log_stdout, yield_lines, parse_model_hypotheses

import torch.distributed as dist
#dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)   # TODO

#print(get_data_preprocessed_dir("en"))


def preprocess_with_fairseq(data_directory,
                            destination_directory,
                            source_lang="src",
                            target_lang="tgt",
                            dataset_implementation="raw",
                            trainpref="train",
                            validpref="valid",
                            testpref="test"):
    #ddir = "/home/AK/skrjanec/toydata/data/"
    full_train_prefix = data_directory / trainpref
    full_valid_prefix = data_directory / validpref
    full_test_prefix = data_directory / testpref

    list_arg = ["--source-lang", source_lang, "--target-lang", target_lang, "--trainpref", full_train_prefix,
                "--validpref", full_valid_prefix, "--testpref", full_test_prefix,
                "--destdir", destination_directory,
                "--dataset-impl", dataset_implementation]

    list_arg = [str(a) for a in list_arg]

    preprocessing_parser = options.get_preprocessing_parser()
    preprocess_args = preprocessing_parser.parse_args(list_arg)
    print("*** Starting preprocessing")
    preprocess.main(preprocess_args)

    """
    This will create dict* files and binary files in the destdir
    Note that overwriting doesn't happen: an error will be raised (FileExistsError) if the command is re-run.
    So empty the dir before?  -> done in fairseq_preprocess_train_generate.py
    """


def train_with_fairseq(dir_with_preprocessed_files, experiment_dir,
                       batch_size=16,
                       lr=0.002,
                       arch="transformer",
                       max_epoch=10,
                       optimizer="adam",
                       dir_checkpoints_suffix="checkpoints",
                       source_lang="src",
                       target_lang="tgt",
                       dataset_implementation="raw"):
    """ Prepare the arguments as a list, pass them to the parser and pass the parser
     to the  train.main(train_args)

     dir_with_preprocessed_files: pointing to the dir with train/val/test.src-tgt.src/tgt and vocab*
     experiment_dir: str or Path, repository_path+experiments+experiment_ID
    """
    # NOTE: the first arg is "data" (no flag?) and it's a dir that has to contain the preprocessed files (dict)
    # as well as ?
    # every
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
    save_dir_full_path = experiment_dir / dir_checkpoints_suffix
    # /home/skrjanec/rewrite_text/experiments/03/checkpoints
    if not os.path.exists(save_dir_full_path):
        os.makedirs(save_dir_full_path)
    mini_args = [dir_with_preprocessed_files, "--arch", arch, "--max-epoch", max_epoch, "--source-lang",
                 source_lang, "--target-lang", target_lang, "--save-dir", save_dir_full_path,
                 "--batch-size", batch_size, "--dataset-impl", dataset_implementation, "--task", "translation",
                 "--optimizer", optimizer, "--lr", lr, "--criterion", "label_smoothed_cross_entropy",
                 "--label-smoothing", 0.54, "--no-epoch-checkpoints"]

    # mini_args = ["/home/AK/skrjanec/toydata/data_bin/", "--arch", "transformer", "--max-epoch", "5", "--source-lang",
    #              source_lang, "--target-lang", target_lang, "--save-dir", "/home/AK/skrjanec/toydata/experiments/01/checkpoints/",
    #              "--batch-size", batch_size, "--dataset-impl", dataset_implementation, "--task", "translation",
    #              "--no-epoch-checkpoints"]

    mini_args = [str(a) for a in mini_args]
    train_parser = options.get_training_parser()
    mini_train_args = options.parse_args_and_arch(train_parser, mini_args)
    print("*** Starting training")
    train.main(mini_train_args)


def generate_with_fairseq(dir_with_model_test_data_and_vocab,
                          dir_with_test_data_and_vocab,
                          batch_size=16,
                          beam_size=8,
                          model_name="checkpoint_best.pt",
                          dataset_implementation="raw",
                          source_vocab_fname="dict.src.txt",
                          target_vocab_fname="dict.tgt.txt",
                          source_test_fname="test.src-tgt.src",
                          target_test_fname="test.src-tgt.tgt"):
    # the first argument is a directory that contains the model, the vocabulary dict* and test files
    # copy the dict* and test* files from respective directories
    #print("dir with model, move the test data and vocabs here", dir_with_model_test_data_and_vocab)
    #print("dir with all data and vocabs, move from here", dir_with_test_data_and_vocab)
    if isinstance(dir_with_model_test_data_and_vocab, str):
        dir_with_model_test_data_and_vocab = Path(dir_with_model_test_data_and_vocab)
    if isinstance(dir_with_test_data_and_vocab, str):
        dir_with_test_data_and_vocab = Path(dir_with_test_data_and_vocab)
    #print("dir with model, move the test data and vocabs here", dir_with_model_test_data_and_vocab)
    #print("dir with all data and vocabs, move from here", dir_with_test_data_and_vocab)
    # copy vocab into the dir_with_model_test_data_and_vocab
    source_vocab_full = dir_with_test_data_and_vocab / source_vocab_fname  # origin
    target_vocab_full = dir_with_test_data_and_vocab / target_vocab_fname  # origin
    dest_src_vocab = dir_with_model_test_data_and_vocab / source_vocab_fname  # destination
    dest_tgt_vocab = dir_with_model_test_data_and_vocab / target_vocab_fname  # destination
    #print("origin and destination, SOURCE VOCAB", source_vocab_full, dest_src_vocab)
    #print("origin and destination, *TARGET* VOCAB", target_vocab_full, dest_tgt_vocab)
    shutil.copy(source_vocab_full, dest_src_vocab)
    shutil.copy(target_vocab_full, dest_tgt_vocab)

    # copy test data into the dir_with_model_test_data_and_vocab
    source_test_full = dir_with_test_data_and_vocab / source_test_fname  # origin
    target_test_full = dir_with_test_data_and_vocab / target_test_fname  # origin
    dest_src_test = dir_with_model_test_data_and_vocab / source_test_fname  # destination
    dest_tgt_test = dir_with_model_test_data_and_vocab / target_test_fname  # destination
    shutil.copy(source_test_full, dest_src_test)
    shutil.copy(target_test_full, dest_tgt_test)

    model_path = dir_with_model_test_data_and_vocab / model_name
    generation_args = [dir_with_model_test_data_and_vocab, "--path", model_path,
                       "--batch-size", batch_size, "--beam", beam_size,
                       "--dataset-impl", dataset_implementation]

    # # path: path to the model
    # generation_arg = ["/home/AK/skrjanec/toydata/experiments/01/checkpoints", "--path",
    #                   "/home/AK/skrjanec/toydata/experiments/01/checkpoints/checkpoint_best.pt",
    #                   "--batch-size", "16", "--beam", "8", "--dataset-impl", dataset_implementation]

    generation_arg = [str(a) for a in generation_args]
    generate_parser = options.get_generation_parser()
    gen_args = options.parse_args_and_arch(generate_parser, generation_arg)

    out_file = str(dir_with_model_test_data_and_vocab / "generation.out")
    print("*** Starting generation - inference")
    with log_stdout(out_file, mute_stdout=True):
        generate.main(gen_args)  # TODO: write this st out into a specific file
        

    # parse this file to fetch out the hypotheses H-N, order them 0, 1, 2... and evaluate with sacrebleu
    ordered_hypotheses = parse_model_hypotheses(out_file)
    # overwrite the file with unordered hypotheses
    out_file2 = str(dir_with_model_test_data_and_vocab / "generation2.out")
    with open(out_file2, "w") as fout:
        for line in ordered_hypotheses:
            fout.write(line[0] + "\n")


def evaluation_automatic_metrics(dir_with_model_test_data_and_vocab,
                                 source_test_fname="test.src-tgt.src",
                                 target_test_fname="test.src-tgt.tgt"):

    # evaluation metrics with EASSE
    # this is not fairseq, but it's easier to add evaluation here
    eval_script = get_evaluation_dir() / "easse_evaluate.sh"
    result_file = open(dir_with_model_test_data_and_vocab / "evaluation.txt", "w")
    dest_src_test = dir_with_model_test_data_and_vocab / source_test_fname
    dest_tgt_test = dir_with_model_test_data_and_vocab / target_test_fname
    out_file2 = str(dir_with_model_test_data_and_vocab / "generation2.out")
 
    commands = ["sh", str(eval_script), str(dest_src_test), str(dest_tgt_test), out_file2]
    result = subprocess.call(commands, stdout=result_file)
    result_file.close()


