"""
Script to create a table with an overview over all training runs, i.e. the
parameters used for training and the evaluation results
"""
import json
import os

from utils.helpers import load_yaml
import argparse
from os import listdir


def create_eval_overview(out_file: str, experiments: list):
    """
    Creates a file with all parameter values and all evaluation metric scores for
    different experiments; one line per experiment; first line includes the column names
    Columns are separated by ';'
    :param experiments: list of the IDs of the experiments to include, str or int
                        if not provided then the results from all experiments in the ../experiments folder are used
    :param out_file: path/name of the output file
    """
    if not experiments:
        exp_ids = [e_id for e_id in listdir('../experiments')]
        experiments = exp_ids

    if isinstance(experiments[0], int):
        experiments = [str(e_id) for e_id in experiments]

    # column names
    header = ['ID', 'Language', 'Features', 'Batch Size', 'Test Batch Size', 'Max Epochs',
               'Beam Size', 'Learning Rate', 'Bleu', 'Sari', 'Fkgl',
               'Bertscore Precision', 'Bertscore Recall', 'Bertscore F1']

    with open(out_file, "w", encoding="utf-8") as result_file:
        result_file.write(';'.join(header))
        result_file.write("\n")

        # one line per experiment
        for config_file in os.listdir('../configs/parameter_tuning'):
            name, ending = config_file.split('.')
            exp_id = ''
            char_ind = -1
            while True:
                try:
                    exp_id += str(int(name[char_ind]))
                    char_ind -= 1
                except:
                    break
            exp = exp_id[::-1]

            if exp not in experiments:
                continue

            # parameters values are specified in the config file
            config_path = f'../configs/parameter_tuning/{name}{exp}.{ending}'
            config_file = load_yaml(config_path)

            # evaluation scores are in the output files from the training
            eval_path = f'../experiments/{exp}/checkpoints/evaluation.txt'

            # make sure that a trained model exists for the current configuration file:
            if exp not in os.listdir('../experiments'):
                print(f'No trained model found for experiment ID {exp}. Configuration file {config_file} will be skipped')
                continue

            with open(eval_path, "r", encoding="utf-8") as eval_f:
                metrics = eval_f.readline()
                metrics = metrics.strip()
                metrics = metrics.replace("'", "\"")
            metric_dict = json.loads(metrics)

            # extract and join all values in the correct order
            exp_row = [exp]
            exp_row.append(config_file['language'])
            features = config_file['features_requested']
            if len(features) == 4:
                feature_abbr = 'all'
            else:
                feature_abbr = ''
                if 'dependency' in features:
                    feature_abbr += 'dep_'
                if 'frequency' in features:
                    feature_abbr += 'freq_'
                if 'length' in features:
                    feature_abbr += 'len_'
                if 'levenshtein' in features:
                    feature_abbr += 'leven'
                if feature_abbr[-1] == '_':
                    feature_abbr = feature_abbr[:-1]
            exp_row.append(feature_abbr)
            exp_row.append(config_file['batch_size'])
            exp_row.append(config_file['test_batch_size'])
            exp_row.append(config_file['max_epochs'])
            exp_row.append(config_file['beam_size'])
            exp_row.append(config_file['lr'])

            exp_row.append(metric_dict['bleu'])
            exp_row.append(metric_dict['sari'])
            exp_row.append(metric_dict['fkgl'])
            exp_row.append(metric_dict['bertscore_precision'])
            exp_row.append(metric_dict['bertscore_recall'])
            exp_row.append(metric_dict['bertscore_f1'])

            exp_row = [str(el) for el in exp_row]

            result_file.write(';'.join(exp_row))
            result_file.write('\n')


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="name for output file")
    parser.add_argument("--exps", required=False, help="list of all experiment IDs to include (int or str)")
    args = vars(parser.parse_args())

    if args["exp"]:
        create_eval_overview(args["out"], args["exp"])
    else:
        create_eval_overview(args["out"], [])




