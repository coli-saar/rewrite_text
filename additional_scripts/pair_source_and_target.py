import argparse


def pair_source_and_target(src_file, tgt_file, output_file, sort_by_length):
    """
    Create a file with source_sentence \tab target_sentence in each line
    for all sentence pairs in the source input file and the corresponding target input file
    If sort_by_length is True then the sentence pairs will be sorted by the length of the
    source sentence in ascending order
    :param src_file: path to source file
    :param tgt_file: path to target file
    :param output_file: path to output file
    :param sort_by_length: whether sentence pairs get sorted by length of source sentence or
                            original order is kept
    """
    source_file = open(src_file, 'r', encoding='utf-8')
    target_file = open(tgt_file, 'r', encoding='utf-8')

    sentence_pairs = []

    for (source_line, target_line) in zip(source_file, target_file):
        sentence_pairs.append((source_line.strip(), target_line.strip()))

    # sort by length of the source sentence
    if sort_by_length:
        sentence_pairs = sorted(sentence_pairs, key = lambda pair: len(pair[0].split(' ')))

    with open(output_file, 'w', encoding='utf-8') as out:

        for (source_sentence, target_sentence) in sentence_pairs:
            out.write(f'{source_sentence}\t{target_sentence}\n')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to source file")
    parser.add_argument("--tgt", required=True, help="Path to corresponding target file")
    parser.add_argument("--out", required=True, help="Path to output file")
    parser.add_argument("--sort", required=False,
                        action='store_true',
                        help="If included then the source-target sentence pairs get sorted by the length of "
                             "the source sentence ")

    args = vars(parser.parse_args())

    sort_by_len = True if args["sort"] else False
    
    pair_source_and_target(args["src"], args["tgt"], args["out"], sort_by_len)

