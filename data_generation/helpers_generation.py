import numpy as np
import shutil
from utils.feature_bin_preparation import create_bins
from utils.feature_extraction import get_bin_value
from utils.helpers import yield_lines, load_tokenizer, run_sentencepiece_tokenizer, run_spacy_tokenizer


def prepare_special_token_string(features_values_dict, feature2token):
    features = sorted(list(features_values_dict.keys()))  # ordered alphabetically
    s = ""
    for f in features:
        temp = "<" + feature2token[f] + "_" + str(features_values_dict[f]) + "> "
        s += temp
    # s is, for example, "<MaxDep_0.3> <FreqRank_0.6> <Leven_0.8> "
    return s


def preprocess_src_file(source_path, destination_path, special_token_str, lang):
    # open the destination file
    destin_source = open(destination_path, "w", encoding="utf-8")
    # load the tokenizer
    tokenizer_model = load_tokenizer('sentpiece', lang)
    # yield lines
    for line in yield_lines(source_path):
        # tokenize line
        tokenized_line = run_sentencepiece_tokenizer(line, tokenizer_model)
        # prepend the special tokens
        p = special_token_str + ' '.join(tokenized_line) + "\n" # has \n at the end
        destin_source.write(p)
    destin_source.close()
    print("Wrote file ", str(destination_path))


def update_requested_features_with_bins(requested_feat_dict):
    feature_bins = create_bins()
    for f_name, f_value in requested_feat_dict.items():
        f_bin = get_bin_value(f_value, feature_bins[f_name])
        requested_feat_dict[f_name] = round(f_bin, 2)
    return requested_feat_dict


def copy_vocab_files(origin_dir, destination_dir,
                     model_name="checkpoint_best.pt",
                     dataset_implementation="raw",
                     source_vocab_fname="dict.src.txt",
                     target_vocab_fname="dict.tgt.txt",):
    """ The vocab files have to be in the same directory as the input file """
    # copy vocab into the dir_with_model_test_data_and_vocab
    source_vocab_full = origin_dir / source_vocab_fname  # origin
    target_vocab_full = origin_dir / target_vocab_fname  # origin
    dest_src_vocab = destination_dir / source_vocab_fname  # destination
    dest_tgt_vocab = destination_dir / target_vocab_fname  # destination
    #print("origin and destination, SOURCE VOCAB", source_vocab_full, dest_src_vocab)
    #print("origin and destination, *TARGET* VOCAB", target_vocab_full, dest_tgt_vocab)
    shutil.copy(source_vocab_full, dest_src_vocab)
    shutil.copy(target_vocab_full, dest_tgt_vocab)
    print("Vocab files copied into the dir that has the input source file")

# fd = {"dependency": 0.3, "frequency": 0.6, "levenshtein": 0.8}
# feature2spec_token = {"dependency": "MaxDep", "frequency": "FreqRank", "length": "Length", "levenshtein": "Leven"}
#
# print(prepare_special_token_string(fd, feature2spec_token))

# req = {"levenshtein": 0.22, "length": 0.78}
# req = (update_requested_features_with_bins(req))
# text = " \n".join([f_name + ": " + str(f_val) for f_name, f_val in req.items()])
# print(text)