import csv
from pathlib import Path
import os
from robustness.tools.imagenet_helpers import ImageNetHierarchy, common_superclass_wnid
from sklearn.model_selection import train_test_split


class OpenSetProtocol:
    """Version 2 of open set protocols"""

    def __init__(self, imagenet_dir, metadata_path, protocol_num=1):
        """ Constructs an open set protocol.

        Args:
            imagenet_dir: Root directory of imagenet (../ILSVRC2012). It must contain the test and
                            val directories.
            metadata_path: Folder of imagenet metadata: imagenet_class_index.json, wordnet.is_a.txt,
                            words.txt.
            protocol_num: Choose protocol 1, 2 or 3.
        """
        self.imagenet_dir = Path(imagenet_dir)
        self.metadata_path = Path(metadata_path)
        self.hierarchy = ImageNetHierarchy(imagenet_dir, metadata_path)
        self.protocol = protocol_num
        self.data = {}

        if self.protocol == 1:
            self.kn_superclasses = ['n02084071']  # dog
            self.neg_superclasses = [
                'n02118333',  # fox
                'n02115335',  # wild_dog
                'n02114100',  # wolf
                'n02120997',  # feline
                'n02131653',  # bear
                'n02441326',  # musteline
                'n02370806',  # ungulate
                'n02469914',  # primate
            ]
            self.unk_superclasses = [
                'n07555863',  # food
                'n03791235',  # motor_vehicle
                'n03183080',  # device
            ]

        elif self.protocol == 2:
            self.kn_superclasses = ['n02087122']  # hunting_dog
            self.neg_superclasses = self.kn_superclasses
            self.unk_superclasses = [
                'n02085374',  # toy_dog
                'n02118333',  # fox
                'n02115335',  # wild_dog
                'n02114100',  # wolf
                'n02120997',  # feline
                'n02131653',  # bear
                'n02441326',  # musteline mammal
                'n02370806',  # ungulate
            ]
        elif self.protocol == 3:
            self.kn_superclasses = common_superclass_wnid('mixed_13')
            # 'mixed_13':
            #              'n02084071',  # dog,
            #              'n01503061',  # bird (52)
            #              'n02159955',  # insect (27)
            #              'n03405725',  # furniture (21)
            #              'n02512053',  # fish (16),
            #              'n02484322',  # monkey (13)
            #              'n02958343',  # car (10)
            #              'n02120997',  # feline (8),
            #              'n04490091',  # truck (7)
            #              'n13134947',  # fruit (7)
            #              'n12992868',  # fungus (7)
            #              'n02858304',  # boat (6)
            #              'n03082979',  # computer(6)
            self.neg_superclasses = None
            # unk_superclasses plus subclasses of mixed_13, appended in update_classes()
            self.unk_superclasses = [
                'n01661091',  # reptile
                'n03051540',  # clothing
                'n02370806',  # ungulate
                'n07707451',  # vegetable
                'n02686568',  # aircraft
            ]
        else:
            raise Exception("Choose between [1,2,3]")

        self.kn_classes = []
        self.neg_classes = []
        self.unk_classes = []
        self.label_map = None

    def get_descendants_wid(self, node_wn_id, in_imagenet=True):
        """ Returns a sorted list of descendants of a class in the wordnet structure.
        Taken from: https://github.com/MadryLab/robustness

        Args:
            node_wn_id(str): Wordnet id of the parent sunset.
            in_imagenet(bool): If True, only considers descendants among ImageNet synsets,
                        else considers all possible descendants in the WordNet hierarchy.

        Returns:
            Sorted list of descendants of node_wn_id.
        """
        if in_imagenet:
            return sorted([ww for ww in self.hierarchy.tree[node_wn_id].descendants_all
                           if ww in set(self.hierarchy.in_wnids)])
        else:
            return sorted(list(self.hierarchy.tree[node_wn_id].descendants_all))

    @staticmethod
    def read_directory(dir_path: Path):
        """ Returns sorted list of files inside the directory.

        Args:
            dir_path(Path): directory to read.

        Returns:
            images: list of images inside directory
            parents: list of repeated parent directory, used to label images
        """
        file_list = sorted(dir_path.glob('*'))
        images = ['/'.join(p.parts[-3:]) for p in file_list]
        parents = [dir_path.name] * len(images)
        return images, parents

    def update_classes(self):
        """ Updates the lists of kn_classes, neg_classes and unk_classes by querying all descendants
        of the selected superclasses. Then the classes are filtered according to protocol design.
        Removes possible duplicates inside the lists and creates a dictionary to map class-label.
        """
        if self.protocol == 1:
            for super_id in self.kn_superclasses:
                self.kn_classes.extend(self.get_descendants_wid(super_id))
            for super_id in self.neg_superclasses:
                self.neg_classes.extend(self.get_descendants_wid(super_id))
            for super_id in self.unk_superclasses:
                self.unk_classes.extend(self.get_descendants_wid(super_id))

        elif self.protocol == 2:
            all_descendants = []
            for super_id in self.kn_superclasses:
                all_descendants.extend(self.get_descendants_wid(super_id))
            # Shuffles the descendant list and selects half for kn and half for negatives
            # random.Random(seed).shuffle(all_descendants)
            middle = len(all_descendants) // 2
            self.kn_classes.extend(all_descendants[:middle])
            self.neg_classes.extend(all_descendants[middle:])
            for super_id in self.unk_superclasses:
                self.unk_classes.extend(self.get_descendants_wid(super_id))

        elif self.protocol == 3:
            for super_id in self.kn_superclasses:
                descendants = self.get_descendants_wid(super_id)
                for idx, class_ in enumerate(descendants):
                    if idx % 2 == 0:
                        self.kn_classes.append(class_)
                    elif idx % 2 != 0 and idx % 3 == 0:
                        self.unk_classes.append(class_)
                    elif idx % 2 != 0 and idx % 3 != 0:
                        self.neg_classes.append(class_)
            for super_id in self.unk_superclasses:
                self.unk_classes.extend(self.get_descendants_wid(super_id))

        # Remove duplicates and sort the classes
        self.kn_classes = sorted(list(set(self.kn_classes)))
        self.neg_classes = sorted(list(set(self.neg_classes)))
        self.unk_classes = sorted(list(set(self.unk_classes)))
        # Dictionary with class label
        self.label_map = dict(zip(self.kn_classes, range(len(self.kn_classes))))

    def query_images(self, target_classes, imagenet_split):
        """ Reads the imagenet directory structure, returns a tuple of images_paths and class_name.

        Args:
            target_classes(list): List of classes to query images.
            imagenet_split(str): Possible values:'train', 'val'. The splits in imagenet dataset.

        Returns:
            images: list of all paths of images in all target_classes.
            classes: list of class ids.
        """
        if imagenet_split not in ['train', 'val']:
            raise Exception('Imagenet data should be in train or val directory')

        images, classes = [], []
        for curr_class in target_classes:
            curr_dir = self.imagenet_dir / imagenet_split / curr_class
            im_list, im_class = self.read_directory(curr_dir)
            images.extend(im_list)
            classes.extend(im_class)
        return images, classes

    def get_label(self, class_name):
        """ Maps the label of a class name
        Args:
            class_name(str): Class name to get an integer label
        Returns:
            Integer label
        """
        if class_name in self.kn_classes:
            return self.label_map[class_name]
        elif class_name in self.neg_classes:
            return -1
        else:
            return -2

    @staticmethod
    def save_csv(path, data):
        """ Simple csv file saver

        Args:
            path(Path): File path
            data(Iterable): Data iterable
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding="utf-8") as file_:
            writer = csv.writer(file_)
            writer.writerows(data)

    def save_datasets_to_csv(self, out_dir):
        """ Saves train, validation and test datasets in separated csv files.

        Args:
            out_dir(Path): Directory to save file.
        """
        out_dir = Path(out_dir)
        # write csv files
        self.save_csv(out_dir / (f'p{self.protocol}_train.csv'), self.data['train'])
        self.save_csv(out_dir / (f'p{self.protocol}_val.csv'), self.data['val'])
        self.save_csv(out_dir / (f'p{self.protocol}_test.csv'), self.data['test'])
        print("Protocol files saved in " + str(out_dir))

    def create_dataset(self, random_state=42):
        """ Create the datasets of the protocol.
        Args:
            random_state(int): Integer.
        """
        self.update_classes()

        # query all images
        images, classes = self.query_images(
            target_classes=[*self.kn_classes, *self.neg_classes],
            imagenet_split='train')

        # create train and validation splits
        x_train, x_val, y_train, y_val = train_test_split(
            images,
            classes,
            train_size=0.8,
            stratify=classes,
            random_state=random_state)

        y_train = [self.get_label(c) for c in y_train]
        y_val = [self.get_label(c) for c in y_val]
        self.data['train'] = list(zip(x_train, y_train))
        self.data['val'] = list(zip(x_val, y_val))

        # test data
        images, classes = self.query_images(
            target_classes=[*self.kn_classes, *self.neg_classes, *self.unk_classes],
            imagenet_split='val'
        )
        classes = [self.get_label(p) for p in classes]
        self.data['test'] = list(zip(images, classes))

    def print_data(self):
        """ Prints general information about the protocol"""

        print(f"\nProtocol {self.protocol}")
        print("-----------------Training Data-----------")
        print(f"Known classes: {len(self.kn_classes)}")
        print(f'Negative classes: {len(self.neg_classes)}')
        print(f"Train dataset size: {len(self.data['train'])}")
        print("-----------------Validation Data---------")
        print(f"Known classes: {len(self.kn_classes)}")
        print(f"Negative classes: {len(self.neg_classes)}")
        print(f"Validation dataset size: {len(self.data['val'])}")
        print("-----------------Test Data---------------")
        print(f"Known classes: {len(self.kn_classes)}")
        print(f"Negative classes: {len(self.neg_classes)}")
        print(f"Unknown classes: {len(self.unk_classes)} \n")
