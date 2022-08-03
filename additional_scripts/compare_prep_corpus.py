"""
Function to find and count all sentences for which characters are "lost" during the
sentence piece tokenization, i.e. where decoding the tokenized sentence after the
feature preprocessing does not look exactly the same as the original not tokenized sentence.

Can also be used to check whether the file created by merging preprocessed shards yields
a file with exactly the same sentences in the same order as the original file contained.
"""

from utils.helpers import load_tokenizer, run_sentencepiece_tokenizer, run_spacy_tokenizer
import argparse
from pathlib import Path
from spacy.tokens import Doc


# TODO: explain the two different files that get created
# TODO: run with spacy tokenizer to check that it works as expected
def compare_preprocessed_vs_original(language: str, tokenizer_type:  str):
    """
    Compares the sentences in the original data set in language 'language' to the corresponding sentences
    in the preprocessed data set with all four features.
    Creates a file "not_match.txt" in the current directory with all sentences that are not equivalent.
    If not all splits should be checked or not all 4 features were extracted the variables 'folder_prep'
    and 'files' need to be adapted accordingly below
    :param language: 'de' or 'en' (files from feature preprocessing need to exist already)
    :param tokenizer_type: 'spacy' or 'sentpiece'
    """
    folder_orig = f'data/{language}/'
    folder_prep = f'data_preprocessed/{language}/dependency_frequency_length_levenshtein/'
    files = ["train.src", "train.tgt", "test.src", "test.tgt", "valid.src", "valid.tgt"]

    characters_dropped = './different_after_tokenization.txt'
    other_differences = './different_after_preprocessing.txt'
    if Path(characters_dropped).exists() or Path(other_differences).exists():
        print("Warning: At least of of the files './different_after_tokenization.txt' and './different_after_preprocessing.txt' "
              "does already exist. Please remove or rename the files and run again.")
        return

    tokenizer = load_tokenizer(tokenizer_type, language)

    not_matching_lines = 0
    not_matching_lines_cleaned = 0

    for comp_file in files:
        orig_file = folder_orig + comp_file
        prep_file = folder_prep + comp_file

        orig = open(orig_file, "r", encoding="utf-8")
        prep = open(prep_file, "r", encoding="utf-8")

        for i, (orig_line, prep_line) in enumerate(zip(orig, prep)):
            orig_line = orig_line.strip()

            prep_line = prep_line.strip()
            prep_line = prep_line.split(" ")
            if comp_file[-1] == "c":
                # first four elements are the features in the .src files
                prep_line = prep_line[4:]

            # Texts from the original corpus get first tokenized and then decoded again
            # in order to ignore white spaces that were removed during tokenization when
            # comparing the texts
            if tokenizer_type == 'sentpiece':
                encoded_orig_line = run_sentencepiece_tokenizer(orig_line, tokenizer)
                decoded_orig_line = tokenizer.decode(encoded_orig_line)

                decoded_text = tokenizer.decode(prep_line)

            else:
                encoded_orig_line = run_spacy_tokenizer(orig_line, tokenizer)
                decoded_orig_line = Doc(tokenizer.vocab, words=encoded_orig_line).text

                decoded_text = Doc(tokenizer.vocab, words=prep_line).text

            orig_tokens = decoded_orig_line.split(" ")
            decoded_tokens = decoded_text.split(" ")

            # check whether lines are missing or have a different order after preprocessing
            if not orig_tokens == decoded_tokens:
                not_matching_lines_cleaned += 1
                print("Found non matching lines:")
                print(comp_file)
                print(str(i+1))
                print(orig_tokens)
                print(decoded_tokens)
                with open(other_differences, "a", encoding="utf-8") as nm:
                    nm.write(comp_file)
                    nm.write("\n")
                    nm.write(orig_line)
                    nm.write("\n")
                    nm.write(decoded_text)
                    nm.write("\n")

            # check for which sentences the tokenization dropped characters / changed the text
            if not orig_line.split(" ") == decoded_tokens:
                not_matching_lines += 1
                with open(characters_dropped, "a", encoding="utf-8") as nm:
                    nm.write(comp_file)
                    nm.write("\n")
                    nm.write(orig_line)
                    nm.write("\n")
                    nm.write(decoded_text)
                    nm.write("\n")

        orig.close()
        prep.close()

    print(f'Not matching lines: {not_matching_lines}')
    print(f'Not matching lines after cleaning: {not_matching_lines_cleaned}')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--l", required=True, help="language of the corpus")
    parser.add_argument("--tokenizer", required=True, help="type of tokenizer that was used for preprocessing, 'spacy' or 'sentpiece'")
    args = vars(parser.parse_args())
    compare_preprocessed_vs_original(args["l"], args["tokenizer"])
