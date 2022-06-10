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
# directory where ImageNet data is present
root_dir = '/local/scratch/datasets/ImageNet/ILSVRC2012'
# create a directory to store files for Protocol 2
out_dir = '/local/scratch/palechor/openset-imagenet/data/old'

# create object to browse ImageNet hierarchy
in_hier = ImageNetHierarchy(in_path, in_info_path)

# ====================================================================================================================#
# Protocol 2:
# Known classes - Some classes of a dog subclass
# Known unknown classes - Some other classes of the same dog subclass
# Unknown unknown classes - Some classes of another dog subclass and some other 4-legged animal classes
# ====================================================================================================================#

# Create Known, known unknown and unknown unknown dictionaries for Protocol 2

# Known classes - some classes of Hunting Dog (since this dog subclass has the highest number of ImageNet descendants)

# manually search WordNet ID for Hunting Dog in "ImageNet_Superclasses.txt"
hunting_dog = 'n02087122'
# initialize empty dictionary for classes of Hunting Dog
hunting_dog_dict = {}
# iterate through WordNet IDs, sorted according to the number of descendants they have (in_hier.wnid_sorted) 
for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(in_hier.wnid_sorted):
    # check if WordNet ID = WordNet ID for Hunting Dog
    if wnid == hunting_dog:
        # iterate through all hunting dog subclasses
        for cnt, wnid in enumerate(in_hier.tree[hunting_dog].descendants_all):
            # check if subclass is a ImageNet Class
            if wnid in in_hier.in_wnids:
                # add subclass to dictionary for classes of Hunting Dog
                hunting_dog_dict[wnid] = in_hier.wnid_to_name[wnid]

# Select the first half of the Hunting Dog dictionary elements as known classes
known_knowns_p2 = dict(list(hunting_dog_dict.items())[len(hunting_dog_dict) // 2:])
# print number of known classes
print('Number of known known classes for Protocol 2:', len(known_knowns_p2), flush=True)

# Known unknown classes - Some other classes of Hunting Dog

# Select the remaining of Hunting Dog dictionary elements as known unknown classes
known_unknowns_p2 = dict(list(hunting_dog_dict.items())[:len(hunting_dog_dict) // 2])
# print number of known unknown classes
print('Number of known unknown classes for Protocol 2:', len(known_unknowns_p2))

# Unknown unknown classes - Some classes of another dog subclass and some other 4-legged animal classes

# Another dog subclass = Toy dog
# manually search WordNet ID for Toy Dog in "ImageNet_Superclasses.txt"
toy_dog = 'n02085374'
# initialize empty dictionary for unknown unknown classes
unknown_unknowns_p2 = {}
# iterate through WordNet IDs, sorted according to the number of descendants they have 
for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(in_hier.wnid_sorted):
    # check if WordNet ID = WordNet ID for Toy Dog
    if wnid == toy_dog:
        # iterate through all toy dog subclasses
        for cnt, wnid in enumerate(in_hier.tree[toy_dog].descendants_all):
            # check if subclass is a ImageNet Class
            if wnid in in_hier.in_wnids:
                # add subclass to dictionary created for unknown unknown classes
                unknown_unknowns_p2[wnid] = in_hier.wnid_to_name[wnid]

# Some some 4-legged animal classes
# fox = 'n02118333'
# wild_dog = 'n02115335'
# wolf = 'n02114100'
# feline = 'n02120997'
# bear = 'n02131653'
# musteline mammal = 'n02441326'
# ungulate = 'n02370806'

# manually search WordNet IDs for some 4-legged animal classes in "ImageNet_Superclasses.txt" 
# and "https://observablehq.com/@mbostock/imagenet-hierarchy"
animal_superclass_wnid = ['n02118333', 'n02115335', 'n02114100', 'n02120997', 'n02131653', 'n02441326',
                          'n02370806']
# initialize empty dictionary for 4-legged animal classes
other4legged = {}
# iterate through all subclasses of the selected WordNet IDs
for c in animal_superclass_wnid:
    for cnt, wnid in enumerate(in_hier.tree[c].descendants_all):
        # check if subclass is a ImageNet Class
        if wnid in in_hier.in_wnids:
            # add subclass to dictionary created for 4-legged animal classes
            other4legged[wnid] = in_hier.wnid_to_name[wnid]

# add 4-legged animal classes to dictionary created for unknown unknown classes
unknown_unknowns_p2.update(other4legged)
# print number of unknown unknown classes
print('Number of unknown unknown classes for Protocol 2:', len(unknown_unknowns_p2))

# create lists with only the keys (WordNet IDs) from dictionaries created for known, known unknown and 
# unknown unknown classes 
# list of known WordNet IDs
kk_p2 = list(known_knowns_p2.keys())
# list of known unknown WordNet IDs
ku_p2 = list(known_unknowns_p2.keys())
# list of unknown unknown WordNet IDs
uu_p2 = list(unknown_unknowns_p2.keys())

# # reconfirm length of kk (known), ku (known unknown) and uu (unknown unknown) lists
# print (len (kk_p2))
# print (len (ku_p2))
# print (len (uu_p2))

# # check that there is no overlap between kk, ku and uu lists
# print (len (set (kk_p2).intersection (set (ku_p2))))
# print (len (set (kk_p2).intersection (set (uu_p2))))
# print (len (set (ku_p2).intersection (set (uu_p2))))

# ====================================================================================================================#
# Create csv files for train, validation and test sets using dictionaries for 
# known, known unknown and unknown unknown classes
# ====================================================================================================================#

# 80% of ILSVRC2012 training data is used for training, remaining 20% is used for validation
percentage_of_training_samples = 0.8

# create integer ground truth labels for known classes from 0 to (Number of known classes-1)
class_mappings_kk = dict(zip(kk_p2, range(len(kk_p2))))
# integer ground truth label for known unknown classes = -1
# integer ground truth label for unknown unknown classes = -2

# Create csv for Known training and validation set

# initialize empty list for known training images
kk_p2_training_list = []
# initialize empty list for known validation images
kk_p2_validation_list = []
# iterate through all known WordNet IDs
for cls_name in kk_p2:
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
    kk_p2_training_list.extend(image_rows[:int(len(image_rows) * percentage_of_training_samples)])
    # add the remaining 20% of the above list to the list of known validation images
    kk_p2_validation_list.extend(image_rows[int(len(image_rows) * percentage_of_training_samples):])

# print length of known training list
print(F"Length of kk_p2_training_list: {len(kk_p2_training_list)}")
# print length of known validation list
print(F"Length of kk_p2_validation_list: {len(kk_p2_validation_list)}")

# create csv file using known training list and store it in the directory created for Protocol 2
with open(f"{out_dir}/train_kkp2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p2_training_list)
# create csv file using known validation list and store it in the directory created for Protocol 2
with open(f"{out_dir}/val_kkp2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p2_validation_list)

# Create csv for Known unknown training and validation set

# initialize empty list for known unknown training images
ku_p2_training_list = []
# initialize empty list for known unknown validation images
ku_p2_validation_list = []
# iterate through all known unknown WordNet IDs
for cls_name in ku_p2:
    # get the entire path of training images in root directory
    image_names = glob(f"{root_dir}/train/{cls_name}/*")
    # from the above path, extract only the path of the training images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and 
    # -1 ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=-1))
    # add 80% of the above list to the list of known unknown training images
    ku_p2_training_list.extend(image_rows[:int(len(image_rows) * percentage_of_training_samples)])
    # add the remaining 20% of the above list to the list of known unknown validation images
    ku_p2_validation_list.extend(image_rows[int(len(image_rows) * percentage_of_training_samples):])

# print length of known unknown training list
print(F"Length of ku_p2_training_list: {len(ku_p2_training_list)}")
# print length of known unknown validation list
print(F"Length of ku_p2_validation_list: {len(ku_p2_validation_list)}")

# create csv file using known unknown training list and store it in the directory created for Protocol 2
with open(f"{out_dir}/train_kup2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ku_p2_training_list)
# create csv file using known unknown validation list and store it in the directory created for Protocol 2
with open(f"{out_dir}/val_kup2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ku_p2_validation_list)

# Create csv for known, known unknowns and unknown unknowns test set
# Test set is created using the entire ILSVRC2012 validation set

# initialize empty list for known test images
kk_p2_test_list = []
# iterate through all known WordNet IDs
for cls_name in kk_p2:
    # get the entire path of test images in root directory
    image_names = glob(f"{root_dir}/val/{cls_name}/*")
    # from the above path, extract only the path of the test images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and 
    # the corresponding known class ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=class_mappings_kk[cls_name]))
    # add the above list to the list of known test images
    kk_p2_test_list.extend(image_rows)

# print length of known test list
print(F"Length of kk_p2_test_list: {len(kk_p2_test_list)}")

# create csv file using known test list and store it in the directory created for Protocol 2
with open(f"{out_dir}/test_kkp2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p2_test_list)

# initialize empty list for known unknown test images
ku_p2_test_list = []
# iterate through all known unknown WordNet IDs
for cls_name in ku_p2:
    # get the entire path of test images in root directory
    image_names = glob(f"{root_dir}/val/{cls_name}/*")
    # from the above path, extract only the path of the test images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and 
    # -1 ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=-1))
    # add the above list to the list of known unknown test images
    ku_p2_test_list.extend(image_rows)

# print length of known unknown test list
print(F"Length of ku_p2_test_list: {len(ku_p2_test_list)}")

# create csv file using known unknown test list and store it in the directory created for Protocol 2
with open(f"{out_dir}/test_kup2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ku_p2_test_list)

# initialize empty list for unknown unknown test images
uu_p2_test_list = []
# iterate through all unknown unknown WordNet IDs
for cls_name in uu_p2:
    # get the entire path of test images in root directory
    image_names = glob(f"{root_dir}/val/{cls_name}/*")
    # from the above path, extract only the path of the test images in the corresponding class folders
    image_names = [i.split(f"{root_dir}/")[-1] for i in image_names]
    # create a list of tuple elements, with each tuple element having the above image path and 
    # -2 ground truth integer label
    image_rows = list(zip_longest(image_names, [], fillvalue=-2))
    # add the above list to the list of unknown unknown test images
    uu_p2_test_list.extend(image_rows)

# print length of unknown unknown test list
print(F"Length of uu_p2_test_list: {len(uu_p2_test_list)}")

# create csv file using unknown unknown test list and store it in the directory created for Protocol 2
with open(f"{out_dir}/test_uup2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(uu_p2_test_list)

# merge known and known unknown training lists to create a combined training csv file 
# and store it in the directory created for Protocol 2
with open(f"{out_dir}/train_p2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p2_training_list + ku_p2_training_list)

# merge known and known unknown validation lists to create a combined validation csv file 
# and store it in the directory created for Protocol 2
with open(f"{out_dir}/val_p2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p2_validation_list + ku_p2_validation_list)

# merge known, known unknown and unknown unknown test lists to create a combined test csv file 
# and store it in the directory created for Protocol 2
with open(f"{out_dir}/test_p2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(kk_p2_test_list + ku_p2_test_list + uu_p2_test_list)

# print length of combined training, validation and test sets
print(F"P2 - Length of training list: {len(kk_p2_training_list + ku_p2_training_list)}")
print(F"P2 - Length of validation list: {len(kk_p2_validation_list + ku_p2_validation_list)}")
print(F"P2 - Length of test list: {len(kk_p2_test_list + ku_p2_test_list + uu_p2_test_list)}")
