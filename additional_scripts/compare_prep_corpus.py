"""
Function to find and count all sentences for which characters are "lost" during the
sentence piece tokenization, i.e. where decoding the tokenized sentence after the
feature preprocessing does not look exactly the same as the original not tokenized sentence.
"""

from utils.helpers import load_tokenizer
import argparse

# TODO: explain the two different files that get created
def compare_preprocessed_vs_original(language):
    """
    Compares the sentences in the original data set in language 'language' to the corresponding sentences
    in the preprocessed data set with all four features.
    Creates a file "not_match.txt" in the current directory with all sentences that are not equivalent.
    If not all splits should be checked or not all 4 features were extracted the variables 'folder_prep'
    and 'files' need to be adapted accordingly below
    :param language: 'de' or 'en' (files from feature preprocessing need to exist already)
    """
    folder_orig = f'data/{language}/'
    folder_prep = f'data_preprocessed/{language}/dependency_frequency_length_levenshtein_complete/'
    files = ["train.src", "train.tgt", "test.src", "test.tgt", "valid.src", "valid.tgt"]

    tokenizer = load_tokenizer("sentpiece", "de")

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

            encoded_orig_line = tokenizer.encode_as_pieces(orig_line)
            decoded_orig_line = tokenizer.decode(encoded_orig_line)

            if comp_file[-1] == "c":
                # first four elements are the features
                prep_line = prep_line[4:]

            decoded_text = tokenizer.decode(prep_line)

            orig_tokens = decoded_orig_line.split(" ")
            decoded_tokens = decoded_text.split(" ")

            if not orig_tokens == decoded_tokens:
                not_matching_lines_cleaned += 1
                print("Found non matching lines:")
                print(comp_file)
                print(str(i+1))
                print(orig_tokens)
                print(decoded_tokens)
                with open("./not_match.txt", "a", encoding="utf-8") as nm:
                    nm.write(orig_line)
                    nm.write("\n")
                    nm.write(decoded_text)
                    nm.write("\n")

            # check for which sentences the tokenization dropped characters / changed the text
            if not orig_line.split(" ") == decoded_tokens:
                not_matching_lines += 1
                with open("./not_match_before_tokenization.txt", "a", encoding="utf-8") as nm:
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
    args = vars(parser.parse_args())
    compare_preprocessed_vs_original(args["l"])

