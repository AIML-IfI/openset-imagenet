# OpenSet protocols V2
from robustness.tools.imagenet_helpers import ImageNetHierarchy, common_superclass_wnid
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import csv


class OpenSetProtocol:

    def __init__(self, im_root_dir, info_path, out_dir, protocol=1):
        self.im_root_dir = Path(im_root_dir)
        self.info_path = Path(info_path)
        self.out_dir = Path(out_dir)
        self.hierarchy = ImageNetHierarchy(im_root_dir, info_path)
        self.protocol = protocol
        self.data = {}
        if self.protocol == 1:
            self.kn_superclasses = ['n02084071']
            self.kn_unk_superclasses = ['n02118333', 'n02115335', 'n02114100', 'n02120997',
                                        'n02131653', 'n02441326', 'n02370806', 'n02469914']
            self.unk_unk_superclasses = ['n07555863', 'n03791235', 'n03183080']

        elif self.protocol == 2:
            self.kn_superclasses = ['n02087122']
            self.kn_unk_superclasses = self.kn_superclasses  # Split the subclasses
            self.unk_unk_superclasses = ['n02085374', 'n02118333', 'n02115335', 'n02114100',
                                         'n02120997', 'n02131653', 'n02441326', 'n02370806']
        elif self.protocol == 3:
            self.kn_superclasses = common_superclass_wnid('mixed_13')
            # Defined inside the mixed_13 classes
            self.kn_unk_superclasses = None
            # Selected classes + subclasses of mixed_13, selected in update_classes()
            self.unk_unk_superclasses = ['n01661091', 'n03051540', 'n02370806', 'n07707451', 'n02686568']

        else:
            raise Exception("Choose between [1,2,3]")

        self.kn_classes = []
        self.kn_unk_classes = []
        self.unk_unk_classes = []
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
        """Updates the lists of kn_classes, kn_unk_classes and unk_unk_classes by querying all descendants
        of the selected superclasses. Then the classes are filtered according to protocol design. Removes
        possible duplicates inside the lists and creates a dictionary to map class-label."""
        if self.protocol == 1:
            for super_c_id in self.kn_superclasses:
                self.kn_classes.extend(self.get_descendants_wid(super_c_id))
            for super_c_id in self.kn_unk_superclasses:
                self.kn_unk_classes.extend(self.get_descendants_wid(super_c_id))
            for super_c_id in self.unk_unk_superclasses:
                self.unk_unk_classes.extend(self.get_descendants_wid(super_c_id))

        elif self.protocol == 2:
            all_descendants = []
            for super_c_id in self.kn_superclasses:
                all_descendants.extend(self.get_descendants_wid(super_c_id))
            # TODO: current selection is half-half for kn and kn_unk, should it be random?
            middle = len(all_descendants) // 2
            self.kn_classes.extend(all_descendants[:middle])
            self.kn_unk_classes.extend(all_descendants[middle:])
            for super_c_id in self.unk_unk_superclasses:
                self.unk_unk_classes.extend(self.get_descendants_wid(super_c_id))

        elif self.protocol == 3:
            for super_c_id in self.kn_superclasses:
                descendants = self.get_descendants_wid(super_c_id)
                for idx, class_ in enumerate(descendants):
                    if idx % 2 == 0:
                        self.kn_classes.append(class_)
                    elif idx % 2 != 0 and idx % 3 == 0:
                        self.unk_unk_classes.append(class_)
                    elif idx % 2 != 0 and idx % 3 != 0:
                        self.kn_unk_classes.append(class_)
            for super_c_id in self.unk_unk_superclasses:
                self.unk_unk_classes.extend(self.get_descendants_wid(super_c_id))

        # Remove duplicates
        self.kn_classes = sorted(list(set(self.kn_classes)))
        self.kn_unk_classes = sorted(list(set(self.kn_unk_classes)))
        self.unk_unk_classes = sorted(list(set(self.unk_unk_classes)))
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
        elif class_name in self.kn_unk_classes:
            return -1
        else:
            return -2

    @staticmethod
    def save_csv(path, data):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def create_dataset(self, random_state=42):
        """ Create the datasets of the protocol
        Args:
            random_state: Integer
        """
        self.update_classes()

        # train - val protocol 1
        x, y = self.query_images([*self.kn_classes, *self.kn_unk_classes], imagenet_split='train')
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, stratify=y, random_state=random_state)
        y_train = [self.get_label(c) for c in y_train]
        y_val = [self.get_label(c) for c in y_val]
        self.data['train'] = list(zip(x_train, y_train))
        self.data['val'] = list(zip(x_val, y_val))

        # test data
        x, y = self.query_images([*self.kn_classes, *self.kn_unk_classes, *self.unk_unk_classes], imagenet_split='val')
        y = [self.get_label(p) for p in y]
        self.data['test'] = list(zip(x, y))

        # write csv files
        self.save_csv(self.out_dir / ('p'+str(self.protocol)+'_train'), self.data['train'])
        self.save_csv(self.out_dir / ('p'+str(self.protocol)+'_val'), self.data['val'])
        self.save_csv(self.out_dir / ('p'+str(self.protocol)+'_test'), self.data['test'])

# ----------------------------------------------------------------------------------------------

in_info_path = Path(r"/local/scratch/datasets/ImageNet/ILSVRC2012/robustness")
root_dir = Path(r"/local/scratch/datasets/ImageNet/ILSVRC2012")
out_dir = Path(r"/local/scratch/palechor/openset-imagenet/data")

prt = OpenSetProtocol(im_root_dir=root_dir, info_path=in_info_path, out_dir=out_dir, protocol=1)
prt.create_dataset(random_state=4242)

print('kn classes:', len(prt.kn_classes))
print('kn_unk classes:', len(prt.kn_unk_classes))
print('unk_unk classes:', len(prt.unk_unk_classes))
print('train size:', len(prt.data['train']))
print('val size:', len(prt.data['val']))
print('test size:', len(prt.data['test']))
''
# def check_datasets(d1, d2):
#     d1 = d1.copy()
#     d2 = d2.copy()
#     print('d1 shape:', d1.shape)
#     print('d2 shape:', d2.shape)
#
#     d1['folder'] = d1.path.apply(lambda x: x.split('/')[1])
#     d1 = (d1.groupby('folder').count())
#     d1 = d1.drop(columns=['path'])
#     d2['folder'] = d2.path.apply(lambda x: x.split('/')[1])
#     d2 = (d2.groupby('folder').count())
#     d2 = d2.drop(columns=['path'])
#     print('same classes:', set(d1.index) == set(d2.index))
#
#     df = pd.merge(d1, d2, on='folder', how='outer')
#     print(df[abs(df.label_x - df.label_y) == 1])
#
#
# in_info_path = Path(r"/local/scratch/datasets/ImageNet/ILSVRC2012/robustness")
# root_dir = Path(r"/local/scratch/datasets/ImageNet/ILSVRC2012")
#
# prt = OpenSetProtocol(im_root_dir=root_dir, info_path=in_info_path, protocol=3)
# prt.create_dataset(random_state=1232)
# prt.write_csv(path='data/p3_train.csv', data=prt.data['train'])
# df1 = pd.DataFrame(prt.data['train'], columns=['path', 'label'])
# df2 = pd.read_csv('data/p3_2/train_p3.csv', names=['path', 'label'])
# print('========== Protocol - Train ==========')
# check_datasets(df1, df2)

# prt.create_dataset(random_state=1232)
# df1 = pd.DataFrame(prt.data['val'], columns=['path', 'label'])
# df2 = pd.read_csv('data/p3_2/val_p3.csv', names=['path', 'label'])
# print('========== Protocol - Val ==========')
# check_datasets(df1, df2)
#
# prt.create_dataset(random_state=1232)
# df1 = pd.DataFrame(prt.data['test'], columns=['path', 'label'])
# df2 = pd.read_csv('data/p3_2/test_p3.csv', names=['path', 'label'])
# print('========== Protocol - Test ==========')
# check_datasets(df1, df2)


