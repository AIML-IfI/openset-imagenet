# OpenSet protocols V2
from robustness.tools.imagenet_helpers import ImageNetHierarchy, common_superclass_wnid
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv


class OpenSetProtocol:

    def __init__(self, im_root_dir, info_path, protocol=1):
        self.im_root_dir = Path(im_root_dir)
        self.info_path = Path(info_path)
        self.hierarchy = ImageNetHierarchy(im_root_dir, info_path)
        self.protocol = protocol
        self.data = {}

        if self.protocol == 1:
            self.kn_superclasses = ['n02084071']  # dog
            self.neg_superclasses = ['n02118333',  # fox
                                     'n02115335',  # wild_dog
                                     'n02114100',  # wolf
                                     'n02120997',  # feline
                                     'n02131653',  # bear
                                     'n02441326',  # musteline
                                     'n02370806',  # ungulate
                                     'n02469914',  # primate
                                     ]
            self.unk_superclasses = ['n07555863',  # food
                                     'n03791235',  # motor_vehicle
                                     'n03183080',  # device
                                     ]

        elif self.protocol == 2:
            self.kn_superclasses = ['n02087122']  # hunting_dog
            self.neg_superclasses = self.kn_superclasses  # Split the subclasses
            self.unk_superclasses = ['n02085374',  # toy_dog
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
            # 'mixed_13': ['n02084071',  # dog,
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
            # Defined inside the mixed_13 classes
            self.neg_superclasses = None
            # Selected classes + subclasses of mixed_13, selected in update_classes()
            self.unk_superclasses = ['n01661091',  # reptile
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
        Args:
            node_wn_id: class wordnet id
            in_imagenet: If true, it returns only classes in the imagenet dataset
        Returns:
            Sorted list of descendants of node_wn_id
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
            dir_path: directory to read.

        Returns: Sorted files list
        """
        file_lst = sorted(dir_path.glob('*'))
        images = ['/'.join(p.parts[-3:]) for p in file_lst]
        parents = [dir_path.name] * len(images)
        return images, parents

    def update_classes(self):
        """
        Updates the lists of kn_classes, neg_classes and unk_classes by querying all
        descendants of the selected superclasses. Then the classes are filtered according to
        protocol design. Removes possible duplicates inside the lists and creates a dictionary
        to map class-label.
        """
        if self.protocol == 1:
            for super_c_id in self.kn_superclasses:
                self.kn_classes.extend(self.get_descendants_wid(super_c_id))
            for super_c_id in self.neg_superclasses:
                self.neg_classes.extend(self.get_descendants_wid(super_c_id))
            for super_c_id in self.unk_superclasses:
                self.unk_classes.extend(self.get_descendants_wid(super_c_id))

        elif self.protocol == 2:
            all_descendants = []
            for super_c_id in self.kn_superclasses:
                all_descendants.extend(self.get_descendants_wid(super_c_id))
            # TODO: current selection is half-half for kn and kn_unk, should it be random?
            middle = len(all_descendants) // 2
            self.kn_classes.extend(all_descendants[:middle])
            self.neg_classes.extend(all_descendants[middle:])
            for super_c_id in self.unk_superclasses:
                self.unk_classes.extend(self.get_descendants_wid(super_c_id))

        elif self.protocol == 3:
            for super_c_id in self.kn_superclasses:
                descendants = self.get_descendants_wid(super_c_id)
                for idx, class_ in enumerate(descendants):
                    if idx % 2 == 0:
                        self.kn_classes.append(class_)
                    elif idx % 2 != 0 and idx % 3 == 0:
                        self.unk_classes.append(class_)
                    elif idx % 2 != 0 and idx % 3 != 0:
                        self.neg_classes.append(class_)
            for super_c_id in self.unk_superclasses:
                self.unk_classes.extend(self.get_descendants_wid(super_c_id))

        # Remove duplicates and sort the classes
        self.kn_classes = sorted(list(set(self.kn_classes)))
        self.neg_classes = sorted(list(set(self.neg_classes)))
        self.unk_classes = sorted(list(set(self.unk_classes)))
        # Dictionary with class label
        self.label_map = dict(zip(self.kn_classes, range(len(self.kn_classes))))

    def query_images(self, target_classes, imagenet_split: str):
        """Reads the imagenet directory structure, returns a tuple of images paths and class name
        Args:
            target_classes: List of classes of whom images are queried.
            imagenet_split: Possible values:'train', 'val'. The splits in imagenet dataset
        Returns: Tuple with list of images paths and class name.
        """
        if imagenet_split not in ['train', 'val']:
            raise Exception('Imagenet data should be in train or val directory')

        images, classes = [], []
        for curr_class in target_classes:
            curr_dir = self.im_root_dir / imagenet_split / curr_class
            im_list, im_class = self.read_directory(curr_dir)
            images.extend(im_list)
            classes.extend(im_class)
        return images, classes

    def get_label(self, class_name):
        """ Maps the label of a class name
        Args:
            class_name: Class name to get an integer label
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
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def save_datasets_to_csv(self, out_dir):
        out_dir = Path(out_dir)
        # write csv files
        self.save_csv(self.out_dir / ('p' + str(self.protocol) + '_train.csv'), self.data['train'])
        self.save_csv(self.out_dir / ('p' + str(self.protocol) + '_val.csv'), self.data['val'])
        self.save_csv(self.out_dir / ('p' + str(self.protocol) + '_test.csv'), self.data['test'])
        print("Protocol files saved in " + str(out_dir))

    def create_dataset(self, random_state=42):
        """ Create the datasets of the protocol
        Args:
            random_state: Integer
        """
        self.update_classes()

        # query all images
        x, y = self.query_images(
            [*self.kn_classes, *self.neg_classes],
            imagenet_split='train'
        )
        
        # create train and validation splits
        x_train, x_val, y_train, y_val = train_test_split(
            x, y,
            train_size=0.8,
            stratify=y,
            random_state=random_state
        )

        y_train = [self.get_label(c) for c in y_train]
        y_val = [self.get_label(c) for c in y_val]
        self.data['train'] = list(zip(x_train, y_train))
        self.data['val'] = list(zip(x_val, y_val))

        # test data
        x, y = self.query_images(
            [*self.kn_classes, *self.neg_classes, *self.unk_classes],
            imagenet_split='val'
        )
        y = [self.get_label(p) for p in y]
        self.data['test'] = list(zip(x, y))
