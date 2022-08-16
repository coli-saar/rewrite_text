from pathlib import Path
import argparse
from utils.paths import configs_dir

# Parameters that are kept the same for all training runs
PREPROCESS = True
TRAIN = True
GENERATE = True
FEATURES_REQUESTED = ["dependency", "frequency", "length", "levenshtein"]
LANGUAGE = 'de'
ARCH = '"transformer"'
OPTIMIZER = '"adam"'
BATCH_SIZE = 16     # max batch size working on the jones-X without memory issues
UPDATE_FREQ = 4     # to get an effective batch size of 64
TEST_BATCH_SIZE = 16
MAX_EPOCHS = 90
PATIENCE = 10
BEAM_SIZE = 8


def create_configs_lr(n_configs: int, start_lr: float, step_size: float, start_id=0, sh_script=False):
    """
    creates n_configs config files for training with the configurations from above
    and each with a different learning rate in the folder configs/parameter_tuning
    first config file hast learning rate start_lr and for each next config file the
    learning rate gets increased by step_size
    :param n_configs: number of config files to create
    :param start_lr: the first, i.e. smallest, learning rate
    :param step_size: the amount by which the learning rate is increased
    :param start_id: the experiment id for the first created config file, the ids for the other
                    experiments are created by increasing start_id by 1 for each file
                    Note!: if a configuration file with the experiment ID already exists it
                            gets overwritten!
    :param sh_script: whether an .sh script should be created that can be launched to run all training configurations
                      after each other and save logs
    """
    path_for_configs = f'{configs_dir}/parameter_tuning/'       # relative to current script
    Path(path_for_configs).mkdir(parents=True, exist_ok=True)
    path_for_configs_run = f'{configs_dir}/parameter_tuning/'  # needs to be relative to the fairseq_preprocess_train_generate.py script

    lr_values = []
    lines_for_sh = []

    for step, exp_id in enumerate(range(start_id, start_id + n_configs)):
        config_name = f'preprocess_train_generate_de{exp_id}.yaml'

        lr_value = start_lr + step * step_size
        lr_value = round(lr_value, 6)
        lr_values.append(lr_value)

        with open(path_for_configs + config_name, "w", encoding="utf-8") as conf_file:
            conf_file.write(f'preprocess: {PREPROCESS}\n')
            conf_file.write(f'train: {TRAIN}\n')
            conf_file.write(f'generate: {GENERATE}\n')
            conf_file.write(f'experiment_id: {exp_id}\n')
            conf_file.write(f'features_requested: {FEATURES_REQUESTED}\n')
            conf_file.write(f'language: {LANGUAGE}\n')
            conf_file.write(f'arch: {ARCH}\n')
            conf_file.write(f'optimizer: {OPTIMIZER}\n')
            conf_file.write(f'batch_size: {BATCH_SIZE}\n')
            conf_file.write(f'update_freq: {UPDATE_FREQ}\n')
            conf_file.write(f'test_batch_size: {TEST_BATCH_SIZE}\n')
            conf_file.write(f'max_epochs: {MAX_EPOCHS}\n')
            conf_file.write(f'patience: {PATIENCE}\n')
            conf_file.write(f'beam_size: {BEAM_SIZE}\n')
            conf_file.write(f'lr: {lr_value}\n')

        if sh_script:
            line = 'python fairseq_preprocess_train_generate.py --config '
            line += path_for_configs_run
            line += config_name
            line += f' |& tee training_logs/training_log{exp_id}.txt\n'
            lines_for_sh.append(line)

    if sh_script:
        print('Created sh script')
        with open("../run_parameter_tuning.sh", "w", encoding="utf-8") as sh:
            for line in lines_for_sh:
                sh.write(line)

    print(f'Created {n_configs} configuration files with the following learning rate values: \n')
    print(f'{lr_values}')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", required=True, help="number of configuration files to create")
    parser.add_argument("--lr", required=True, help="smallest learning rate")
    parser.add_argument("--step", required=True, help="step size for increasing the learning rate for each configuration")
    parser.add_argument("--run", required=False, help="set to True if a shell script should be created to run all "
                                                      "configurations")
    parser.add_argument("--exp", required=False, help="experiment id for the first config file, default is 0")

    args = vars(parser.parse_args())
    if args["exp"] and args["run"]:
        if args["run"] == "True":
            r = True
        else:
            r = False
        create_configs_lr(n_configs=int(args["n"]), start_lr=float(args["lr"]), step_size=float(args["step"]), start_id=int(args["exp"]), sh_script=r)
    elif args["exp"]:
        create_configs_lr(n_configs=int(args["n"]), start_lr=float(args["lr"]), step_size=float(args["step"]), start_id=int(args["exp"]))
    elif args["run"]:
        if args["run"] == "True":
            r = True
        else:
            r = False
        create_configs_lr(n_configs=int(args["n"]), start_lr=float(args["lr"]), step_size=float(args["step"]), sh_script=r)
    else:
        create_configs_lr(n_configs=int(args["n"]), start_lr=float(args["lr"]), step_size=float(args["step"]))
