# import libraries
from itertools import zip_longest
from glob import glob
import csv
import random

random.seed(0)

# ====================================================================================================================#
# Access ImageNet Hierarchy using robustness library
from robustness.tools.imagenet_helpers import ImageNetHierarchy

# set path where data is present
in_path = '/local/scratch/datasets/ImageNet/ILSVRC2012'

# set path where imagenet files (wordnet.is_a.txt, words.txt and imagenet_class_index.json) are present
in_info_path = "/local/scratch/datasets/ImageNet/ILSVRC2012/robustness"

# create object to browse ImageNet hierarchy
in_hier = ImageNetHierarchy(in_path, in_info_path)

# ====================================================================================================================#
# Protocol 3:
# Known classes - Some classes of some ancestors
# Known unknown classes - other classes of the same ancestors
# Unknown unknown classes - some classes of known ancestors and some classes of ancestors other than known ancestors
# ====================================================================================================================#

# Create Known, known unknown and unknown unknown dictionaries for Protocol 3

# import pre-packaged ImageNet subset of "mixed_13" from robustness library
from robustness.tools.imagenet_helpers import common_superclass_wnid

# get WordNet IDs of mixed_13 classes
mixed13_superclass_wnid = common_superclass_wnid('mixed_13')

# 'mixed_13': ['n02084071', #dog,
#             'n01503061', #bird (52)
#             'n02159955', #insect (27)
#             'n03405725', #furniture (21)
#             'n02512053', #fish (16),
#             'n02484322', #monkey (13)
#             'n02958343', #car (10)
#             'n02120997', #feline (8),
#             'n04490091', #truck (7)
#             'n13134947', #fruit (7)
#             'n12992868', #fungus (7)
#             'n02858304', #boat (6)  
#             'n03082979', #computer(6)

# initialize empty dictionary for known classes
known_knowns_p3 = {}
# initialize empty dictionary for known unknown classes
known_unknowns_p3 = {}
# initialize empty dictionary for unknown unknown classes - some classes of known ancestors
unknown_unknowns_p3_v2 = {}
# iterate through all subclasses of the selected WordNet IDs
# print(mixed13_superclass_wnid)
for c in mixed13_superclass_wnid:
    for cnt, wnid in enumerate(in_hier.tree[c].descendants_all):
        # select WordNet IDs at even indices as known classes
        if wnid in in_hier.in_wnids and cnt % 2 == 0:
            known_knowns_p3[wnid] = in_hier.wnid_to_name[wnid]
        # select WordNet IDs at odd indices, which are divisible by 3, as unknown unknown classes
        elif wnid in in_hier.in_wnids and cnt % 2 != 0 and cnt % 3 == 0:
            unknown_unknowns_p3_v2[wnid] = in_hier.wnid_to_name[wnid]
        # select WordNet IDs at remaining odd indices as known unknown classes
        elif wnid in in_hier.in_wnids and cnt % 2 != 0 and cnt % 3 != 0:
            known_unknowns_p3[wnid] = in_hier.wnid_to_name[wnid]

# print(known_knowns_p3)
# print(known_unknowns_p3)
# print(unknown_unknowns_p3_v2)

# print number of known classes
print('Number of known known classes for Protocol 3:', len(known_knowns_p3), flush=True)
# print number of known unknown classes
print('Number of known unknown classes for Protocol 3:', len(known_unknowns_p3))
# print number of unknown unknown classes - some classes of known ancestors
print('Number of unknown unknown classes, that are subclasses of known ancestors for Protocol 3:',
      len(unknown_unknowns_p3_v2))

# Unknown unknown classes - Some classes of other ancestors
# reptile = 'n01661091'
# clothing = 'n03051540'
# ungulate = 'n02370806'
# vegetable = 'n07707451'
# aircraft = 'n02686568'

# manually search WordNet IDs for some classes of other ancestors in "ImageNet_Superclasses.txt"
# and "https://observablehq.com/@mbostock/imagenet-hierarchy"
otherans_superclass_wnid = ['n01661091', 'n03051540', 'n02370806', 'n07707451', 'n02686568']
# initialize empty dictionary for unknown unknown classes
unknown_unknowns_p3 = {}
# iterate through all subclasses of the selected WordNet IDs
for c in otherans_superclass_wnid:
    for cnt, wnid in enumerate(in_hier.tree[c].descendants_all):
        # check if subclass is a ImageNet Class
        if wnid in in_hier.in_wnids:
            # add subclass to dictionary for unknown unknown classes
            unknown_unknowns_p3[wnid] = in_hier.wnid_to_name[wnid]

# add unknown unknown classes of known ancestors to dictionary created for unknown unknown classes
unknown_unknowns_p3.update(unknown_unknowns_p3_v2)
# print number of unknown unknown classes
print('Number of unknown unknown classes for Protocol 3:', len(unknown_unknowns_p3))

# create lists with only the keys (WordNet IDs) from dictionaries created for known, known unknown and
# unknown unknown classes
# list of known WordNet IDs
kk_p3 = list(known_knowns_p3.keys())
# list of known unknown WordNet IDs
ku_p3 = list(known_unknowns_p3.keys())
# list of unknown unknown WordNet IDs
uu_p3 = list(unknown_unknowns_p3.keys())

# # reconfirm length of kk (known), ku (known unknown) and uu (unknown unknown) lists
# print (len (kk_p3))
# print (len (ku_p3))
# print (len (uu_p3))
#
# # check that there is no overlap between kk, ku and uu lists
# print (len (set (kk_p3).intersection (set (ku_p3))))
# print (len (set (kk_p3).intersection (set (uu_p3))))
# print (len (set (ku_p3).intersection (set (uu_p3))))

# ====================================================================================================================#
# Create csv files for train, validation and test sets using dictionaries for
# known, known unknown and unknown unknown classes
# ====================================================================================================================#

# directory where ImageNet data is present
root_dir = '/local/scratch/datasets/ImageNet/ILSVRC2012'
# create a directory to store files for Protocol 3
out_dir = '/local/scratch/palechor/openset-imagenet/data/old'
# 80% of ILSVRC2012 training data is used for training, remaining 20% is used for validation
percentage_of_training_samples = 0.8

# create integer ground truth labels for known classes from 0 to (Number of known classes-1)
class_mappings_kk = dict(zip(kk_p3, range(len(kk_p3))))
# integer ground truth label for known unknown classes = -1
# integer ground truth label for unknown unknown classes = -2

# Create csv for Known known training and validation set

# initialize empty list for known training images
kk_p3_training_list = []
# initialize empty list for known validation images
kk_p3_validation_list = []
# iterate through all known WordNet IDs
for cls_name in kk_p3:
    # get the entire path of training images in root directory
    # eg., '/local/scratch/datasets/ImageNet/ILSVRC2012/train/n02113799/n02113799_78.JPEG'
    image_names = glob(f"{root_dir}/train/{cls_name}/*")
    # from the above path, extract only the path of the training images in the corresponding class folders
    # eg., 'n02113799/n02113799_78.JPEG'
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and
    # the corresponding known class ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=class_mappings_kk[cls_name]))
    # add 80% of the above list to the list of known training images
    kk_p3_training_list.extend(image_rows[:int(len(image_rows) * percentage_of_training_samples)])
    # add the remaining 20% of the above list to the list of known validation images
    kk_p3_validation_list.extend(image_rows[int(len(image_rows) * percentage_of_training_samples):])

# print length of known training list
print(F"Length of kk_p3_training_list: {len(kk_p3_training_list)}")
# print length of known validation list
print(F"Length of kk_p3_validation_list: {len(kk_p3_validation_list)}")

# create csv file using known training list and store it in the directory created for Protocol 3
with open(f"{out_dir}/train_kkp3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p3_training_list)
# create csv file using known validation list and store it in the directory created for Protocol 3
with open(f"{out_dir}/val_kkp3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p3_validation_list)

# Create csv for Known unknown training and validation set

# initialize empty list for known unknown training images
ku_p3_training_list = []
# initialize empty list for known unknown validation images
ku_p3_validation_list = []
# iterate through all known unknown WordNet IDs
for cls_name in ku_p3:
    # get the entire path of training images in root directory
    image_names = glob(f"{root_dir}/train/{cls_name}/*")
    # from the above path, extract only the path of the training images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and
    # -1 ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=-1))
    # add 80% of the above list to the list of known unknown training images
    ku_p3_training_list.extend(image_rows[:int(len(image_rows) * percentage_of_training_samples)])
    # add the remaining 20% of the above list to the list of known unknown validation images
    ku_p3_validation_list.extend(image_rows[int(len(image_rows) * percentage_of_training_samples):])

# print length of known unknown training list
print(F"Length of ku_p3_training_list: {len(ku_p3_training_list)}")
# print length of known unknown validation list
print(F"Length of ku_p3_validation_list: {len(ku_p3_validation_list)}")

# create csv file using known unknown training list and store it in the directory created for Protocol 3
with open(f"{out_dir}/train_kup3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ku_p3_training_list)
# create csv file using known unknown validation list and store it in the directory created for Protocol 3
with open(f"{out_dir}/val_kup3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ku_p3_validation_list)

# Create csv for known knowns, known unknowns and unknown unknowns test set
# Test set is created using the entire ILSVRC2012 validation set

# initialize empty list for known test images
kk_p3_test_list = []
# iterate through all known WordNet IDs
for cls_name in kk_p3:
    # get the entire path of test images in root directory
    image_names = glob(f"{root_dir}/val/{cls_name}/*")
    # from the above path, extract only the path of the test images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and
    # the corresponding known class ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=class_mappings_kk[cls_name]))
    # add the above list to the list of known test images
    kk_p3_test_list.extend(image_rows)

# print length of known test list
print(F"Length of kk_p3_test_list: {len(kk_p3_test_list)}")

# create csv file using known test list and store it in the directory created for Protocol 3
with open(f"{out_dir}/test_kkp3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p3_test_list)

# initialize empty list for known unknown test images
ku_p3_test_list = []
# iterate through all known unknown WordNet IDs
for cls_name in ku_p3:
    # get the entire path of test images in root directory
    image_names = glob(f"{root_dir}/val/{cls_name}/*")
    # from the above path, extract only the path of the test images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and
    # -1 ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=-1))
    # add the above list to the list of known unknown test images
    ku_p3_test_list.extend(image_rows)

# print length of known unknown test list
print(F"Length of ku_p3_test_list: {len(ku_p3_test_list)}")

# create csv file using known unknown test list and store it in the directory created for Protocol 3
with open(f"{out_dir}/test_kup3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ku_p3_test_list)

# initialize empty list for unknown unknown test images
uu_p3_test_list = []
# iterate through all unknown unknown WordNet IDs
for cls_name in uu_p3:
    # get the entire path of test images in root directory
    image_names = glob(f"{root_dir}/val/{cls_name}/*")
    # from the above path, extract only the path of the test images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and
    # -2 ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=-2))
    # add the above list to the list of unknown unknown test images
    uu_p3_test_list.extend(image_rows)

# print length of unknown unknown test list
print(F"Length of uu_p3_test_list: {len(uu_p3_test_list)}")

# create csv file using unknown unknown test list and store it in the directory created for Protocol 3
with open(f"{out_dir}/test_uup3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(uu_p3_test_list)

# merge known and known unknown training lists to create a combined training csv file
# and store it in the directory created for Protocol 3
with open(f"{out_dir}/train_p3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p3_training_list + ku_p3_training_list)

# merge known and known unknown validation lists to create a combined validation csv file
# and store it in the directory created for Protocol 3
with open(f"{out_dir}/val_p3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p3_validation_list + ku_p3_validation_list)

# merge known, known unknown and unknown unknown test lists to create a combined test csv file
# and store it in the directory created for Protocol 3
with open(f"{out_dir}/test_p3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p3_test_list + ku_p3_test_list + uu_p3_test_list)

# print length of combined training, validation and test sets
print(F"P3 - Length of training list: {len(kk_p3_training_list + ku_p3_training_list)}")
print(F"P3 - Length of validation list: {len(kk_p3_validation_list + ku_p3_validation_list)}")
print(F"P3 - Length of test list: {len(kk_p3_test_list + ku_p3_test_list + uu_p3_test_list)}")