import warnings
import sys

warnings.filterwarnings('ignore')
from exp_preparation import *
from bias_mitigation import *

def display_help():
    print("USAGE\n\tpython3 Astral.py [ CONFIGURATION FOLDER ]\n", file=sys.stderr)
    print("Your configurations should be in json format.", file=sys.stderr) 
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        display_help()
    repository = sys.argv[1]
    liste_config = get_config(repository)
    for config in liste_config:
        FL_parameters, data_preparation_info, learning_info, data_info, bias_mitigation_info, remark = config[0], \
                                                                                                       config[1], \
                                                                                                       config[2], \
                                                                                                       config[3], \
                                                                                                       config[4], \
                                                                                                       config[5]
        nb_users = FL_parameters['nb_clients']
        train_sets, validation_set, test_set, warmup_sets = prepare_data(nb_users, data_preparation_info, data_info, FL_parameters)
        initial_model = prepare_model(learning_info, FL_parameters)
        args = {
            "FL_parameters": FL_parameters,
            "train_sets": train_sets,
            "validation_set": validation_set,
            "test_set": test_set,
            "warmup_sets": warmup_sets,
            "FL_model": initial_model,
            "learning_info": learning_info,
            "data_preparation_info": data_preparation_info,
            "data_info": data_info,
            "BiasMitigation_info": bias_mitigation_info,
            "remark": remark
        }
        if bias_mitigation_info["apply"] == 1:
            bias_mitigation = {"ASTRAL_OPT": ASTRAL_OPT_FL}
            bias_mitigation_name = bias_mitigation_info["bias_mitigation_name"]
            if bias_mitigation_name not in bias_mitigation.keys():
                print(f"Bias technique {bias_mitigation_name} not supported.")
            bias_mitigation[bias_mitigation_name](args).run()
        else:
            FLBase(args).run()
    print('===================================================')
    print('Done.')
    print('===================================================')
