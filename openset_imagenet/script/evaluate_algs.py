""" Independent code for inference in testing dataset. The functions are included and executed
in the train.py script."""
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from vast.tools import set_device_gpu, set_device_cpu, device
from torchvision import transforms as tf
from torch.utils.data import DataLoader
from vast import opensetAlgos
from openset_imagenet import approaches
from openset_imagenet import openset_algos
import openset_imagenet
import pickle
from openset_imagenet.openmax_evm import compute_adjust_probs, evm_hyperparams, openmax_hyperparams, compute_probs, compose_dicts, get_ccr_at_fpr, validate

def get_args():
    """Gets the evaluation parameters."""
    parser = argparse.ArgumentParser("Get parameters for evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "configuration",
        type = Path,
        help = "The configuration file that defines the experiment"
    )

    # directory parameters
    parser.add_argument(
        "loss",
        choices = ["entropic", "softmax", "garbage"],
        help="Which loss function to evaluate"
    )
    parser.add_argument(
        "protocol",
        type = int,
        choices = (1,2,3),
        help = "Which protocol to evaluate"
    )

    parser.add_argument(
        "--algorithm", "-alg",
        choices = ["threshold", "openmax", "proser", "evm"],
        help = "Which algorithm to evaluate. Specific parameters should be in the yaml file"
    )

    parser.add_argument(
        "--use-best", "-b",
        action="store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
    )
    parser.add_argument(
        "--gpu", "-g",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
    )
    parser.add_argument(
        "--imagenet-directory",
        type=Path,
        default=Path("/local/scratch/datasets/ImageNet/ILSVRC2012/"),
        help="Imagenet root directory"
    )
    parser.add_argument(
        "--protocol-directory",
        type=Path,
        default = "protocols",
        help = "Where are the protocol files stored"
    )
    parser.add_argument(
        "--output-directory",
        default = "experiments/Protocol_{}",
        help = "Where to find the results of the experiments"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Select the batch size for the test set batches")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Data loaders number of workers, default:4")

    args = parser.parse_args()
    try:
        args.output_directory = args.output_directory.format(args.protocol)
    except:
        pass
    args.output_directory = Path(args.output_directory)
    return args




def main(command_line_options = None):

    args = get_args()
    #args = get_args(command_line_options)
    cfg = openset_imagenet.util.load_yaml(args.configuration)
    if args.gpu:
        cfg.gpu = args.gpu
    cfg.protocol = args.protocol
    cfg.algorithm.type = args.algorithm 
    cfg.output_directory = args.output_directory
    cfg.loss.type = args.loss
    
    # Create transformations
    transform_val = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    # create datasets
    val_dataset = openset_imagenet.ImagenetDataset(
        csv_file=args.protocol_directory/f"p{args.protocol}_val.csv",
        imagenet_path=args.imagenet_directory,
        transform=transform_val)

    #reorganize labels for the validation set
    val_dataset.re_order_labels()
    
    test_dataset = openset_imagenet.ImagenetDataset(
        csv_file=args.protocol_directory/f"p{args.protocol}_test.csv",
        imagenet_path=args.imagenet_directory,
        transform=transform_val)

    # Info on console
    print("\n========== Data ==========")
    print(f"Val dataset len:{len(val_dataset)}, labels:{val_dataset.label_count}")
    print(f"Test dataset len:{len(test_dataset)}, labels:{test_dataset.label_count}")

    # create data loaders
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    # create device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
    else:
        print("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    if args.loss == "garbage":
        n_classes = val_dataset.label_count # we use one class for the negatives
        val_dataset.replace_negative_label()
    else:
        n_classes = val_dataset.label_count - 1  # number of classes - 1 when training with unknowns

    # create model
    suffix = "_best" if args.use_best else "_curr"
    
    
    if cfg.algorithm.type=='proser':
        base = openset_imagenet.ResNet50(
            fc_layer_dim=n_classes,
            out_features=n_classes,
            logit_bias=False)

        model = openset_imagenet.model.ResNet50Proser(
            resnet_base = base,
            dummy_count = cfg.algorithm.dummy_count,
            fc_layer_dim=n_classes)

        model_path = args.output_directory / (args.loss+ "_" + cfg.algorithm.type  + "_" + str(cfg.epochs)+ "_" + str(cfg.algorithm.dummy_count)+ suffix+".pth")
    else:
        model = openset_imagenet.ResNet50(fc_layer_dim=n_classes, out_features=n_classes, logit_bias=False)
        model_path = args.output_directory / (args.loss + suffix+".pth")

    start_epoch, best_score = openset_imagenet.train.load_checkpoint(model, model_path)
    print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)
    
    if cfg.test_on or cfg.eval_on:
        if cfg.algorithm.type == 'openmax':
            print("reading evm model for the parameter combo")
            model_to_test = cfg.openmax_model_to_test.format(cfg.protocol, cfg.loss.type)
            file_handler = open(model_to_test,'rb')
            model_dict = pickle.load(file_handler)
            alg_hyperparameters=[cfg.algorithm.tailsize, cfg.algorithm.distance_multiplier, cfg.algorithm.translateAmount, cfg.algorithm.distance_metric, cfg.algorithm.alpha_om]
            hyperparams = openmax_hyperparams(*alg_hyperparameters)
        elif cfg.algorithm.type == 'evm':
            print("reading evm model for the parameter combo")
            model_to_test = cfg.evm_model_to_test.format(cfg.protocol, cfg.loss.type)
            file_handler = open(model_to_test,'rb')
            model_dict = pickle.load(file_handler)
            alg_hyperparameters = [cfg.algorithm.tailsize, cfg.algorithm.cover_threshold,cfg.algorithm.distance_multiplier, cfg.algorithm.distance_metric, cfg.algorithm.chunk_size]
            hyperparams = evm_hyperparams(*alg_hyperparameters)
    
    if cfg.test_on:
        # Test Section
        print("getting features and logits for test set .... ")

        if cfg.algorithm.type=='proser':
            gt, logits, features, scores = openset_imagenet.proser.get_arrays_for_proser(
                model=model,
                loader=test_loader
            )        
            suffix = suffix + "_" + str(cfg.algorithm.dummy_count)
        else:
            gt, logits, features, scores = openset_imagenet.train.get_arrays(
                model=model,
                loader=test_loader
            )

        if args.loss=='garbage':
            #biggest_label = np.sort(np.unique(gt))[-1]
            #print(biggest_label)
            #gt[gt==biggest_label] = -1
            logits = logits[:,:-1]
            features = features[:,:-1]
            scores = scores[:,:-1]
            print('shapes for logits, features, and scores after reading and excluding BG', logits.shape, features.shape, scores.shape)

        if cfg.algorithm.type == 'openmax':
            #scores are being adjusted her through openmax alpha
            print("adjusting probabilities for openmax with alpha")
            scores = compute_adjust_probs(gt, logits, features, scores, model_dict, cfg, hyperparams, alpha_index=1)
        elif cfg.algorithm.type=='evm':
            print("computing probabilities for evm")
            scores = compute_probs(gt, logits, features, scores, model_dict, cfg, hyperparams)
        
        file_path = args.output_directory / f"{args.loss}_{cfg.algorithm.type}_test_arr{suffix}.npz"
        np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
        print(f"Target labels, logits, features and scores saved in: {file_path}")

    #Validation of one specific model
    if cfg.eval_on:
        #reset suffix
        suffix = "_best" if args.use_best else "_curr"
        
        print("========== Evaluating ==========")
        print("Validation data:")
        # extracting arrays for validation

        
        if cfg.algorithm.type=='proser':
            gt, logits, features, scores = openset_imagenet.proser.get_arrays_for_proser(
                model=model,
                loader=val_loader
            )
            suffix = suffix + "_" + str(cfg.algorithm.dummy_count)        
        else:
            gt, logits, features, scores = openset_imagenet.train.get_arrays(
                model=model,
                loader=val_loader
            )

        if args.loss=='garbage':
            biggest_label = np.sort(np.unique(gt))[-1]
            print(biggest_label)
            gt[gt==biggest_label] = -1
            logits = logits[:,:-1]
            features = features[:,:-1]
            scores = scores[:,:-1]
            print('shapes for logits, features, and scores after reading and excluding BG', logits.shape, features.shape, scores.shape)


        print('in eval', gt)
    
        if cfg.algorithm.type == 'openmax':
            #scores are being adjusted her through openmax alpha
            print("adjusting probabilities for openmax")
            scores = compute_adjust_probs(gt, logits, features, scores, model_dict, cfg, hyperparams, alpha_index=1)
        elif cfg.algorithm.type=='evm':
            print("computing probabilities for evm")
            scores = compute_probs(gt, logits, features, scores, model_dict, cfg, hyperparams)

        file_path = args.output_directory / f"{args.loss}_{cfg.algorithm.type}_val_arr{suffix}.npz"
        np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
        print(f"Target labels, logits, features and scores saved in: {file_path}")

        #file_path = args.output_directory / f"{args.loss}_val_arr{suffix}.npz"
        #np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)




    #file_path = args.output_directory / f"{args.loss}_{cfg.algorithm.type}_test_arr{suffix}.npz"
    #np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    #print(f"Target labels, logits, features and scores saved in: {file_path}")


    
    if cfg.validate_on:

        print("getting features and logits for validation set .... ")
       
        gt, logits, features, scores = openset_imagenet.train.get_arrays(
            model=model,
            loader=val_loader
        )

        print('last label in validation dataset is: ', gt[-1])
        
        if args.loss=='garbage':
            biggest_label = np.sort(np.unique(gt))[-1]
            print(biggest_label)
            gt[gt==biggest_label] = -1
            logits = logits[:,:-1]
            features = features[:,:-1]
            scores = scores[:,:-1]
            print('After relabeling, last label in validation dataset is: ', gt[-1])
            print('shapes for logits, features, and scores after reading and excluding BG', logits.shape, features.shape, scores.shape)





        print('Validation starting for openmax/evm model')

        if cfg.algorithm.type == 'openmax':
            print("reading openmax models for different parameter combos")
            model_to_test = cfg.openmax_model_to_test.format(cfg.protocol, cfg.loss.type)

            file_handler = open(model_to_test,'rb')
            model_dict = pickle.load(file_handler)
            
            alg_hyperparameters=[cfg.algorithm.tailsize, cfg.algorithm.distance_multiplier, cfg.algorithm.translateAmount, cfg.algorithm.distance_metric, cfg.algorithm.alpha_om]
            hyperparams = openmax_hyperparams(*alg_hyperparameters)
            print(hyperparams)

            for ts in cfg.algorithm.tailsize:
                tailsize = f"{ts:.1f}"
                for dm in cfg.algorithm.distance_multiplier:
                    distance_multiplier = f"{dm:.2f}"
                    print('Model loaded for ', ts, dm)
                    model_to_val = cfg.openmax_model_to_validate.format(cfg.protocol, cfg.loss.type, tailsize, distance_multiplier)
                    file_handler = open(model_to_val,'rb')
                    model_dict = pickle.load(file_handler)
                    print('just before validate')
                    validate(gt, logits, features, scores, model_dict, hyperparams, cfg) 


        elif cfg.algorithm.type == 'evm':
            print("reading evm models for the parameter combos")
            model_to_test = cfg.evm_model_to_test.format(cfg.protocol, cfg.loss.type)

            file_handler = open(model_to_test,'rb')
            model_dict = pickle.load(file_handler)

            alg_hyperparameters = [cfg.algorithm.tailsize, cfg.algorithm.cover_threshold,cfg.algorithm.distance_multiplier, cfg.algorithm.distance_metric, cfg.algorithm.chunk_size]
            hyperparams = evm_hyperparams(*alg_hyperparameters)

            for ts in cfg.algorithm.tailsize:
                for dm in cfg.algorithm.distance_multiplier:
                    distance_multiplier = f"{dm:.2f}"
                    model_to_val = cfg.evm_model_to_validate.format(cfg.protocol, cfg.loss.type, ts, distance_multiplier )
                    file_handler = open(model_to_val,'rb')
                    model_dict = pickle.load(file_handler)
                    print('Model loaded for ', ts, dm)
                    validate(gt, logits, features, scores, model_dict, hyperparams, cfg)
            
            print(hyperparams)
    

   
if __name__=='__main__':
    main()
    print("enters")
