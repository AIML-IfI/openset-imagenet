import argparse
from .. import approaches as ap
from pathlib import Path
import pickle
import torch

"""
##################################################
 README
##################################################

This parser has been designed with the goal of providing a systematic guide for performing the
experiments. The focus of the design lies on the user being clearly aware of the chosen options
at every step of the experiment configuration process. Therefore, optional arguments with the
"required=True" configuration have been used instead of positional arguments. Generally, required
options are considered bad form according to the docs but this option has been opted for
according to the above stated reasons (https://docs.python.org/3/library/argparse.html#required).

"""


def command_line_options(models, model_out_path, oscr_path, log_path, data_path, csv_path, ku_target, uu_target, num_workers, model_in_build_dnn, model_in_use_dnn, dnn_features, hyperparameters_evm, hyperparameters_openmax, approaches, fpr_thresholds, model_extendable, hyperparameters_proser, model_in_extend_dnn, optimizer_params):

    approaches = approaches

    model_out_path.mkdir(parents=True, exist_ok=True)
    oscr_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # root parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        prog=f'Evaluation of Open Set Algorithms'
    )
    # note: parser is designed for use on Ifi server 'rolf'
    parser.add_argument('-prot', '--protocol', type=int, choices=[
                        1, 2, 3], help='protocol to be run (-> provides access information for the associated dataset)', required=True)
    parser.add_argument('-arch', '--architecture', type=str, choices=models,
                        help='architecture of the model to be trained/to be used as underlying DNN', required=True)
    parser.add_argument('-g', '--gpu', type=int, default=0, choices=[
                        0, 1, 2, 3, 4, 5, 6, 7], help='specification of to be used GPU on server \'rolf\'')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='batch size (used in dataloader)')

    parser.set_defaults(model_out_path=model_out_path, oscr_path=oscr_path,
                        log_path=log_path, data_path=data_path, csv_path=csv_path)
    parser.set_defaults(ku_target=ku_target, uu_target=uu_target)
    parser.set_defaults(num_workers=num_workers)

    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands',
                                       help="run '<subcommand> --help' for info on usage", dest='approach_category', required=True)

    # Parser hierarchy for approaches that build a DNN (-> SoftMax, Entropic OSL)
    # ################################################################################

    # node parser (LEVEL 1)
    parser_build_dnn = subparsers.add_parser(
        'build_dnn', help='parent parser for approaches that build a DNN (i.e. {Baseline=SoftMax, EntropicOSL})')

    subparsers_build_dnn = parser_build_dnn.add_subparsers(title='subcommands', description='valid subcommands',
                                                           help="run '<subcommand> --help' for info on usage", dest='procedure', required=True)

    # node parser (LEVEL 2)
    parser_build_dnn_train = subparsers_build_dnn.add_parser(
        'train', help='parent parser for the training procedure of approaches that build a DNN')

    parser_build_dnn_train.add_argument('-opt', '--optimizer', type=str, choices=[
        'Adam', 'SGD'], help='optimizer to be used in the training procedure)', required=True)
    parser_build_dnn_train.add_argument('-opt_params_ok', '--optimizer_parameters_ok', action='store_true',
                                        help='are the parameters for the optimizer correctly specified? Check constants LEARNING_RATE, BETA_1, BETA_2 in file \'settings.py\'', required=True)

    parser_build_dnn_train.add_argument(
        '-e', '--epochs', type=int, help='number of epochs used in training procedure', required=True)
    parser_build_dnn_train.add_argument('-feat_dim', '--feature_dimension', type=int, default=512,
                                        help='output dimensionality of the deep feature layer of the to be trained model')

    subparsers_build_dnn_train = parser_build_dnn_train.add_subparsers(title='subcommands', description='valid subcommands',
                                                                       help="run '<subcommand> --help' for info on usage", dest='approach', required=True)

   # leaf parser
    parser_base_train = subparsers_build_dnn_train.add_parser(
        'base', help='training and evaluation of baseline model (-> Softmax)')
    parser_base_train.set_defaults(train_cls=['kk'], val_cls=['kk'])

    # leaf parser
    parser_entropic_train = subparsers_build_dnn_train.add_parser(
        'entropic', help='training and evaluation of model using entropic open set loss')
    parser_entropic_train.set_defaults(
        train_cls=['kk', 'ku'], val_cls=['kk', 'ku'])

    # node parser (LEVEL 2)
    parser_build_dnn_test = subparsers_build_dnn.add_parser(
        'test', help='parent parser for the testing procedure of approaches that build a DNN')

    parser_build_dnn_test.add_argument('-mdl_pth_ok', '--model_to_test_path_ok',  action='store_true',
                                       help='is the filesystem path to the stored model correctly specified? Check constant MODEL_IN_BUILD_DNN_TEST in file \'settings.py\'', required=True)
    parser_build_dnn_test.set_defaults(model_to_test=model_in_build_dnn)

    subparsers_build_dnn_test = parser_build_dnn_test.add_subparsers(title='subcommands', description='valid subcommands',
                                                                     help="run '<subcommand> --help' for info on usage", dest='approach', required=True)

    # leaf parser
    parser_base_test = subparsers_build_dnn_test.add_parser(
        'base', help='testing of baseline model (-> SoftMax)')
    parser_base_test.set_defaults(train_cls=None, val_cls=None)

    # leaf parser
    parser_entropic_test = subparsers_build_dnn_test.add_parser(
        'entropic', help='testing of model using entropic open set loss')
    parser_entropic_test.set_defaults(
        train_cls=None, val_cls=None)

    # Parser hierarchy for approaches that use a pretrained DNN for feature extraction (-> EVM, OpenMax)
    # ################################################################################

    # node parser (LEVEL 1)
    parser_use_dnn = subparsers.add_parser(
        'use_dnn', help='parent parser for approaches that use a pretrained DNN for extracting features and apply a model on them (i.e. {evm, openmax})')

    parser_use_dnn.add_argument('-feat_net_ok', '--feature_network_ok', action='store_true',
                                help='is the network used for extracting features from the datasets correctly specified? Check constant DNN_FEATURES in file \'settings.py\'', required=True)
    parser_use_dnn.set_defaults(dnn_features=Path(dnn_features))
    parser_use_dnn.add_argument('-load_dict', '--load_data_dict', action='store_true', default=False,
                                help='loading stored data dictionaries (i.e. features/logits) instead of creating them. If serialized dictionary objects are not yet created, do not set this flag')

    subparsers_use_dnn = parser_use_dnn.add_subparsers(title='subcommands', description='valid subcommands',
                                                       help="run '<subcommand> --help' for info on usage", dest='procedure', required=True)
    # node parser (LEVEL 2)
    parser_use_dnn_train = subparsers_use_dnn.add_parser(
        'train', help='parent parser for the training procedure of approaches that use a DNN for feature extraction')
    parser_use_dnn_train.add_argument(
        '-fpr_thrs', '--fpr_thresholds', type=float, nargs='+', default=fpr_thresholds, help='list of false positive rate (fpr) thresholds that are used to select the best model in the validation procedure in a "ccr@fpr" fashion (see Dhamija et al., 2018)')

    subparsers_use_dnn_train = parser_use_dnn_train.add_subparsers(title='subcommands', description='valid subcommands',
                                                                   help="run '<subcommand> --help' for info on usage", dest='approach', required=True)

    # leaf parser
    parser_evm_train = subparsers_use_dnn_train.add_parser(
        'evm', help='training and evaluation of EVM model')
    parser_evm_train.add_argument('-hparams_ok', '--hyperparameters_ok',  action='store_true',
                                  help='are the EVM hyperparameters correctly specified for the current evaluation? Check the section "EVM Hyperparameter Specifications" in file \'settings.py\'', required=True)
    parser_evm_train.set_defaults(hyperparameters=hyperparameters_evm)
    parser_evm_train.add_argument('--train_cls', type=str, required=True, nargs='+',
                                  choices=['kk', 'ku'], help='classes to be used in training, must be subset of {kk, ku}')
    parser_evm_train.set_defaults(val_cls=['kk', 'ku'])

    # leaf parser
    parser_openmax_train = subparsers_use_dnn_train.add_parser(
        'openmax', help='training and evaluation of OpenMax model')
    # note: in it's original form, OpenMax only uses KKCs for training
    parser_openmax_train.add_argument('-hparams_ok', '--hyperparameters_ok',  action='store_true',
                                      help='are the OpenMax hyperparameters correctly specified for the current evaluation? Check the section "OpenMax Hyperparameter Specifications" in file \'settings.py\'', required=True)
    parser_openmax_train.add_argument('-alpha_ok', '--alpha_parameters_ok',  action='store_true',
                                      help='are the alpha parameters correctly specified (used in openmax_alpha() and denote number of revisable classes)? Check the constant OM_ALPHA in file \'settings.py\'', required=True)
    parser_openmax_train.set_defaults(hyperparameters=hyperparameters_openmax)
    parser_openmax_train.set_defaults(train_cls=['kk'], val_cls=['kk', 'ku'])

    # node parser (LEVEL 2)
    parser_use_dnn_test = subparsers_use_dnn.add_parser(
        'test', help='parent parser for the testing procedure of approaches that use a DNN for feature extraction')
    parser_use_dnn_test.add_argument('-mdl_pth_ok', '--model_to_test_path_ok',  action='store_true',
                                     help='is the filesystem path to the stored model correctly specified? Check constant MODEL_IN_USE_DNN_TEST in file \'settings.py\'', required=True)
    parser_use_dnn_test.set_defaults(model_to_test=Path(model_in_use_dnn))

    subparsers_use_dnn_test = parser_use_dnn_test.add_subparsers(title='subcommands', description='valid subcommands',
                                                                       help="run '<subcommand> --help' for info on usage", dest='approach', required=True)

    # leaf parser
    parser_evm_test = subparsers_use_dnn_test.add_parser(
        'evm', help='testing of EVM model')

    parser_evm_test.add_argument('-dist_metr_ok', '--distance_metric_ok',  action='store_true',
                                 help='is the distance metric correctly specified? Check constant EVM_DIST_METRIC in file \'settings.py\'', required=True)
    parser_evm_test.set_defaults(hyperparameters=hyperparameters_evm)
    parser_evm_test.set_defaults(train_cls=None, val_cls=None)

    # leaf parser
    parser_openmax_test = subparsers_use_dnn_test.add_parser(
        'openmax', help='testing of OpenMax model')
    parser_openmax_test.add_argument('-dist_metr_ok', '--distance_metric_ok',  action='store_true',
                                     help='is the distance metric correctly specified? Check constant OM_DIST_METRIC in file \'settings.py\'', required=True)
    parser_openmax_test.add_argument('-alpha_ok', '--alpha_parameters_ok',  action='store_true',
                                     help='is the alpha parameter correctly specified (used in openmax_alpha() and denotes number of revisable classes)? Check the constant OM_ALPHA in file \'settings.py\'', required=True)
    parser_openmax_test.set_defaults(hyperparameters=hyperparameters_openmax)
    parser_openmax_test.set_defaults(train_cls=None, val_cls=None)

    # Parser hierarchy for approaches that extend an existing DNN (-> Proser)
    # ################################################################################

    # node parser (LEVEL 1)
    parser_extend_dnn = subparsers.add_parser(
        'extend_dnn', help='parent parser for approaches that extend an existing DNN (i.e. PROSER)')

    subparsers_extend_dnn = parser_extend_dnn.add_subparsers(title='subcommands', description='valid subcommands',
                                                             help="run '<subcommand> --help' for info on usage", dest='procedure', required=True)

    # node parser (LEVEL 2)
    parser_extend_dnn_train = subparsers_extend_dnn.add_parser(
        'train', help='parent parser for the training procedure of approaches that extend a DNN')

    parser_extend_dnn_train.add_argument(
        '-e', '--epochs', type=int, default=10, help='number of epochs used in training procedure')
    parser_extend_dnn_train.add_argument('-basis_net_ok', '--basis_network_ok', action='store_true',
                                         help='is the pretrained closed set network (that is extended by dummy classifiers) correctly specified? Check constant MODEL_EXTENDABLE in file \'settings.py\'', required=True)
    parser_extend_dnn_train.set_defaults(
        model_extendable=Path(model_extendable))

    subparsers_extend_dnn_train = parser_extend_dnn_train.add_subparsers(title='subcommands', description='valid subcommands',
                                                                         help="run '<subcommand> --help' for info on usage", dest='approach', required=True)
    # leaf parser
    parser_proser_train = subparsers_extend_dnn_train.add_parser(
        'proser', help='training and evaluation of Proser model')

    parser_proser_train.add_argument('-hparams_ok', '--hyperparameters_ok',  action='store_true',
                                     help='are the Proser hyperparameters correctly specified for the current evaluation? Check the section "Proser Hyperparameter Specifications" in file \'settings.py\'', required=True)
    parser_proser_train.set_defaults(
        lambda1=hyperparameters_proser[0], lambda2=hyperparameters_proser[1], alpha=hyperparameters_proser[2])
    parser_proser_train.add_argument('-no_dummies', '--no_dummy_clfs', type=int, nargs='+', default=[1, 2, 3, 5, 10],
                                     help='number of dummy classifiers to be added to the pretrained model')
    parser_proser_train.add_argument('-bias_comp', '--bias_computation', type=str, choices=['True', 'False'],
                                     help='compute bias for dummy logit (=True) or use value of zero (=False)', required=True)
    parser_proser_train.set_defaults(train_cls=['kk'], val_cls=['kk', 'ku'])

    # node parser (LEVEL 2)
    parser_extend_dnn_test = subparsers_extend_dnn.add_parser(
        'test', help='parent parser for the testing procedure of approaches that extend a DNN')

    parser_extend_dnn_test.add_argument('-mdl_pth_ok', '--model_to_test_path_ok',  action='store_true',
                                        help='is the filesystem path to the stored model correctly specified? Check constant MODEL_IN_EXTEND_DNN_TEST in file \'settings.py\'', required=True)
    parser_extend_dnn_test.set_defaults(
        model_to_test=Path(model_in_extend_dnn))

    subparsers_extend_dnn_test = parser_extend_dnn_test.add_subparsers(title='subcommands', description='valid subcommands',
                                                                       help="run '<subcommand> --help' for info on usage", dest='approach', required=True)
    # leaf parser
    parser_proser_test = subparsers_extend_dnn_test.add_parser(
        'proser', help='testing of Proser model')
    parser_proser_test.set_defaults(train_cls=None, val_cls=None)

    # Sanity Checks
    # ################################################################################

    args = parser.parse_args()
    args.architecture = args.architecture + '_Feature'
    args.gpu = "cuda:" + str(args.gpu)

    if args.procedure == 'train':
        args.train_cls = sorted(set(args.train_cls))
        args.val_cls = sorted(set(args.val_cls))

    if args.approach_category == 'build_dnn':
        if args.procedure == 'train':
            assert args.epochs > 0 and args.feature_dimension > 0, 'argument must be a positive integer'

            if args.optimizer == 'Adam':
                args.optimizer = {
                    'opt': 'Adam', 'lr': optimizer_params[0], 'beta_1': optimizer_params[1], 'beta_2': optimizer_params[2]}
            else:
                args.optimizer = {
                    'opt': 'SGD', 'lr': optimizer_params[0], 'beta_1': optimizer_params[1]}

        if args.procedure == 'test':
            model_dict = torch.load(args.model_to_test)

            assert args.protocol == model_dict['instance'][
                'protocol'], 'mismatch between specified CLI-protocol and protocol used in training procedure (-> conflicting data sources)'
            assert approaches[args.approach] == model_dict['approach_train'], 'mismatch between chosen approach and approach that underlies the testable model'
            assert args.architecture == model_dict['instance'][
                'architecture'], 'mismatch between CLI-specified architectures across train and test procedure'

    if args.approach_category == 'use_dnn':
        args.hyperparameters = getattr(ap, f'{args.approach}_hyperparams')(
            *args.hyperparameters)

        assert args.protocol == torch.load(args.dnn_features)[
            'instance']['protocol'], 'mismatch between CLI-specified protocol and protocol used in training the underlying, feature-extracting model (-> conflicting data sources)'

        assert args.architecture == torch.load(args.dnn_features)[
            'instance']['architecture'], 'mismatch between CLI-specified architecture and architecture of underlying, feature-extracting model'

        if args.procedure == 'test':
            file_handler = open(args.model_to_test, 'rb')
            model_dict = pickle.load(file_handler)

            assert args.protocol == model_dict['instance'][
                'protocol'], 'mismatch between CLI-specified protocol and protocol used in training procedure (-> conflicting data sources)'
            assert approaches[args.approach] == model_dict['approach_train'], 'mismatch between chosen approach and approach that underlies the testable model'
            assert args.architecture == model_dict['instance'][
                'architecture'], 'mismatch between CLI-specified architectures across train and test procedure'
            assert Path(
                args.dnn_features).stem == model_dict['instance']['used_dnn'], 'underlying models for feature extraction do not match across train and test procedure'
            assert args.hyperparameters.distance_metric == model_dict[
                'distance_metric'], 'mismatch between distance metric in train and test procedure'

    if args.approach_category == 'extend_dnn':
        if args.procedure == 'train':
            args.batch = 32
            assert 'base' in args.model_extendable.name, 'Proser assumes a well-trained closed set model for extension by dummy classifiers'
            assert args.epochs > 0, 'argument must be a positive integer'

            assert args.protocol == torch.load(args.model_extendable)[
                'instance']['protocol'], 'mismatch between CLI-specified protocol and protocol used in training the basis model (-> conflicting data sources)'
            assert args.architecture == torch.load(args.model_extendable)[
                'instance']['architecture'], 'mismatch between CLI-specified architecture and architecture of basis model'

        if args.approach == 'proser' and args.procedure == 'train':
            assert len(args.no_dummy_clfs) > 0, 'empty list not allowed'

        if args.procedure == 'test':
            file_handler = open(args.model_to_test, 'rb')
            model_dict = torch.load(file_handler)

            assert args.protocol == model_dict['instance'][
                'protocol'], 'mismatch between CLI-specified protocol and protocol used in training procedure (-> conflicting data sources)'
            assert approaches[args.approach] == model_dict['approach_train'], 'mismatch between chosen approach and approach that underlies the testable model'
            assert args.architecture == model_dict['instance'][
                'architecture'], 'mismatch between CLI-specified architectures across train and test procedure'

    return args
