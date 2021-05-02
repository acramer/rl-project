
default_configs = {

    # NOTE: Hyper Parameters (ideally) in order of most used:

    'epochs':            {'default': 1,
                          'flags':   ['--epochs', '-e'],
                          'argparse':{'type':int}},
    'learning_rate':     {'default': 0.001,
                          'flags':   ['--learning_rate', '-L'],
                          'argparse':{'type':float}},
    'batch_size':        {'default': 128,
                          'flags':   ['--batch_size', '-B'],
                          'argparse':{'type':int}},
    'weight_decay':      {'default': 0.0002,
                          'flags':   ['--weight_decay', '-W'],
                          'argparse':{'type':float}},
    'save_interval':     {'default': None,
                          'flags':   ['--save_int', '-S'],
                          'argparse':{'type':int,'help':'Save Interval in epochs.  A value of 0 saves at completion only.  All valid interval values save on completion in model.th in specified directory.'}},

    'step_schedule':     {'default': False,
                          'flags':   ['--step_schedule', '-s'],
                          'argparse':{'action':'store_true','help':'Learning rate decay.'}},
    'cosine':            {'default': False,
                          'flags':   ['--cosine'],
                          'argparse':{'action':'store_true','help':'CosineAnnealing Learning Rate Scheduler.'}},
    'adam':              {'default': False,
                          'flags':   ['--adam', '-a'],
                          'argparse':{'action':'store_true','help':'Adam optimizer used.'}},
    'dropout_rate':      {'default': 0.0,
                          'flags':   ['--dropout_rate', '-D'],
                          'argparse':{'type':float}},

    'architecture':      {'default': 'procedural',
                          'flags':   ['--arch', '-A'],
                          'argparse':{'type':str,
                                      'choices':['procedural','central-q'],
                                      'help':'RL Algo Used "Procedural", "Centralized Q"'}},
    'epsilon':          {'default': 0.4,
                          'flags':   ['--epsilon'],
                          'argparse':{'type':float}},
    'num_ants':          {'default': 10,
                          'flags':   ['--num_ants', '-N'],
                          'argparse':{'type':int}},
    'max_steps':         {'default': 20,
                          'flags':   ['--max_steps', '-E'],
                          'argparse':{'type':int}},

    # NOTE: Hyper Parameters that change the least:

    # NOTE: Training/Evaluation args that don't effect learning algorithm:

    # NOTE: Example, maybe implement
    # 'wandb':             {'default': True,
    #                       'flags':   ['--no_logging', '-n'],
    #                       'argparse':{'action':'store_false','help':'Turns logging on Weights&Bias off.'}},
    'description':       {'default': '',
                          'flags':   ['--des'],
                          'argparse':{'type':str,'help':'Model Description, used in folder and log naming.'}},
    # TODO: Will implement
    # 'mode':              {'default': 'train',
    #                       'flags':   ['--mode'],
    #                       'argparse':{'type':str,'choices':['train','test','verify','predict'],'help':'Program mode.  Options include: "train", "test", "verify", "predict"'}},
    # TODO: Will implement
    # 'architecture':      {'default': 'basic',
    #                       'flags':   ['--arch', '-A'],
    #                       'argparse':{'type':str,
    #                                   'choices':['basic','res_skip','efficient','parallel_channel','helix','flat_coding','multi_loop','dla'],
    #                                   'help':'Network Architecture used in the model.  Options include: "basic", "res_skip", "efficient", "parallel_channel", "helix", "flat_coding", "multi_loop","dla"'}},
    'simulate':         {'default': False,
                          'flags':   ['--simulate','-P'],
                          'argparse':{'action':'store_true','help':'Model training and evaluation runs in silent mode.'}},
    'silent':            {'default': False,
                          'flags':   ['--silent'],
                          'argparse':{'action':'store_true','help':'Model training and evaluation runs in silent mode.'}},


    'load_model_dir':    {'default': None,
                          'flags':   ['--load'],
                          'argparse':{'type':str,'help':'Path of the file the network weights are initialzed with.  If any value is passed, the weights are loaded.'}},
    'save_model_dir':    {'default': 'models',
                          'flags':   ['--model_dir'],
                          'argparse':{'type':str,'help':'Path of the directory the model is saved to.'}},
    'log_directory':     {'default': 'logs',
                          'flags':   ['--log_dir'],
                          'argparse':{'type':str,'help':'Path of the log directory for W&B or TB.'}},

}

def print_configs(configs=None):
    print("\n-------------------------------------------------------------------------------")
    print("| {:^75} |".format('Configuration'))
    print("| {:^23} | {:^23} | {:^23} |".format('Option Descriptions','Option Strings', 'Values Passed In' if configs else 'Default Values'))
    print("-------------------------------------------------------------------------------")
    for k,c in default_configs.items():
        value = c['default']
        if configs: value = getattr(configs, k)
        if value is None: value = 'None'
        if isinstance(value, bool): value = str(value)
        print("| {:<23} | {:<23} | {:<23} |".format(k, ' , '.join(c['flags']), value))
        if configs is None and 'help' in c['argparse'].keys(): print("    Help: {} \n".format(c['argparse']['help']))
    print("-------------------------------------------------------------------------------\n")

def parse_configs():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    # Programatically add default_configs to parser arguments
    for k,c in default_configs.items():
        parser.add_argument(*c['flags'], dest=k, default=c['default'], **c['argparse'])
    parser.add_argument('-h', '--help', dest='help', action='store_true', default=False)
    configs = parser.parse_args()

    return configs

### END CODE HERE
