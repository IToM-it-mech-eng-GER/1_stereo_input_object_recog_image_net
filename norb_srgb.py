import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import logging
from tqdm import tqdm
from os import makedirs
from os.path import join
from os.path import exists
from os import link
from itertools import groupby
import pandas as pd


class SmallNORBExample:

    def __init__(self):
        self.image_lt  = None
        self.image_rt  = None
        self.category  = None
        self.instance  = None
        self.elevation = None
        self.azimuth   = None
        self.lighting  = None

    def __lt__(self, other):
        return self.category < other.category or \
                (self.category == other.category and self.instance < other.instance)

    def show(self, subplots):
        fig, axes = subplots
        fig.suptitle(
            'Category: {:02d} - Instance: {:02d} - Elevation: {:02d} - Azimuth: {:02d} - Lighting: {:02d}'.format(
                self.category, self.instance, self.elevation, self.azimuth, self.lighting))
        axes[0].imshow(self.image_lt, cmap='gray')
        axes[1].imshow(self.image_rt, cmap='gray')

    @property
    def pose(self):
        return np.array([self.elevation, self.azimuth, self.lighting], dtype=np.float32)


class SmallNORBDataset:

    # Number of examples in both train and test set
    n_examples = 24300

    # Categories present in small NORB dataset
    categories = ['animal', 'human', 'airplane', 'truck', 'car']

    def __init__(self, dataset_root='',\
                    csv_dataset='', \
                    csv_master_dataset='', \
                    csv_prepared_set='', \
                    csv_prepared_dirs='', \
                    csv_process_dirs='', \
                    master_default='', \
                    dest_default='', \
                    dest_size=224 \
                ):
        """
        Initialize small NORB dataset wrapper
        
        Parameters
        ----------
        dataset_root: str
            Path to directory where small NORB archives have been extracted.
        """
        self.logger = logging.getLogger(__name__)
        self.dataset_root = dataset_root
        self.csv_dataset = csv_dataset
        self.csv_master_dataset = csv_master_dataset
        self.csv_prepared_set = csv_prepared_set
        self.csv_prepared_dirs = csv_prepared_dirs
        self.csv_process_dirs = csv_process_dirs
        self.master_default = master_default
        self.dest_default = dest_default
        self.dest_size = dest_size # same as for eitel-et-al
        self.initialized  = False

        logging.basicConfig(level=logging.INFO)


        # Store path for each file in small NORB dataset (for compatibility the original filename is kept)
        self.dataset_files = {
            'train': {
                'cat':  join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'),
                'info': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'),
                'dat':  join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
            },
            'test':  {
                'cat':  join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'),
                'info': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'),
                'dat':  join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
            }
        }

        # Initialize both train and test data structures
        self.data = {
            'train': [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)],
            'test':  [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)]
        }

        if exists(self.csv_master_dataset):
            print('Dataset already exported (master data) - no initializing necessary')
            return

        # Fill data structures parsing dataset binary files
        for data_split in ['train', 'test']:
            self._fill_data_structures(data_split)

        self.initialized = True

    def explore_random_examples(self, dataset_split):
        """ Visualize random examples for dataset exploration purposes Parameters
        ---------- dataset_split: str       Dataset split, can be either 'train' or 'test'
        Returns ------- None """
        if self.initialized:
            subplots = plt.subplots(nrows=1, ncols=2)
            #i = 0
            for i in np.random.permutation(SmallNORBDataset.n_examples):
                #i = i+1
                self.data[dataset_split][i].show(subplots)
                plt.waitforbuttonpress()
                #if (i > 10):
                #    break
                plt.cla()

    # export_to_jpg ruf tile_border auf und erzeugt master_dataset
    def export_to_jpg(self):
        """ Export all dataset images to `export_dir` directory
        Parameters ----------
        export_dir: str
            Path to export directory (which is created if nonexistent)     
        Returns ------- XXX """
        export_dir = self.master_default
        # num | azimuth | categ | dest_filename      | elevation | inst_numb | lighting| mastfilepath         |orienta| set
        # 0   | 4       | animal| animal_000000_08_6_4_4.jgp|6   | 8         | 4       |../norb_master_dir/...|left   | train 
        if exists(self.csv_master_dataset):
            print('Dataset already exported (master data)')
            return pd.read_csv(self.csv_master_dataset)   

        if self.initialized:
     
            print('Exporting images to {}...'.format(export_dir), end='', flush=True)
            data = []
            for split_name in ['train', 'test']:
                
                split_dir = join(export_dir, split_name)
                if not exists(split_dir):
                    makedirs(split_dir)

                for i, norb_example in enumerate(self.data[split_name]):

                    category = SmallNORBDataset.categories[norb_example.category]
                    instance = norb_example.instance

                    # image_lt_path = join(split_dir, '{:06d}_{}_{:02d}_lt.jpg'.format(i, category, instance))
                    image_lt_path = join(split_dir, '{}_{:06d}_{:02d}_{}_{}_{}_lt.jpg'.format(category, i, instance,norb_example.elevation,norb_example.azimuth, norb_example.lighting))
                    dest_filename = '{}_{:06d}_{:02d}_{}_{}_{}.jpg'.format(category, i, instance,norb_example.elevation,norb_example.azimuth, norb_example.lighting)
                    data.append({
                        'no': i,
                        'category': category,
                        'instance_number': instance,
                        'elevation': norb_example.elevation,
                        'azimuth': norb_example.azimuth,
                        'lightning': norb_example.lighting,
                        'set': split_name,
                        'orientation': 'left',
                        'dest_filename': dest_filename,
                        'master_filepath': image_lt_path
                    })

                    #image_rt_path = join(split_dir, '{:06d}_{}_{:02d}_rt.jpg'.format(i, category, instance))
                    image_rt_path = join(split_dir, '{}_{:06d}_{:02d}_{}_{}_rt.jpg'.format(category, i, instance,norb_example.elevation,norb_example.azimuth))
                    dest_filename = '{}_{:06d}_{:02d}_{}_{}.jpg'.format(category, i, instance,norb_example.elevation,norb_example.azimuth)
                    data.append({
                        'no': i,
                        'category': category,
                        'instance_number': instance,
                        'elevation': norb_example.elevation,
                        'azimuth': norb_example.azimuth,
                        'lightning': norb_example.lighting,
                        'set': split_name,
                        'orientation': 'right',
                        'dest_filename': dest_filename,
                        'master_filepath': image_rt_path
                    })

                    img_lt = self.tile_border(norb_example.image_lt)
                    img_rt = self.tile_border(norb_example.image_rt)
                    scipy.misc.imsave(image_lt_path, img_lt)
                    scipy.misc.imsave(image_rt_path, img_rt)

                    #scipy.misc.imsave(image_lt_path, norb_example.image_lt)
                    #scipy.misc.imsave(image_rt_path, norb_example.image_rt)
            cur_df = pd.DataFrame(data)
            cur_df.to_csv(self.csv_master_dataset, index=False)
            
            print('Done.')
            return cur_df

    # Tile the image to a destination square image, used in Eitel et al.
    def tile_border(self, rgb_image):         
        dst_size = self.dest_size
        old_height = rgb_image.shape[0]
        old_width = rgb_image.shape[1]

        if old_height > old_width:
            rgb_image = rgb_image.transpose(1, 0, 2)

        height = rgb_image.shape[0]
        width = rgb_image.shape[1]

        new_height = int(height * dst_size * 1.0 / width)
        rgb_image = cv2.resize(rgb_image, (dst_size, new_height))
        tiling_size = int((dst_size - new_height) * 1.0 / 2)

        first_row_matrix = np.tile(rgb_image[0, :, :], (tiling_size, 1, 1)) if len(rgb_image.shape) > 2 \
            else np.tile(rgb_image[0, :], (tiling_size, 1))

        last_row_matrix = np.tile(rgb_image[new_height - 1, :, :], (dst_size - new_height - tiling_size, 1, 1)) \
            if len(rgb_image.shape) > 2 \
            else np.tile(rgb_image[new_height - 1, :], (dst_size - new_height - tiling_size, 1))

        rgb_image = np.concatenate([first_row_matrix,
                                    rgb_image,
                                    last_row_matrix],
                                axis=0)

        if old_height > old_width:
            rgb_image = rgb_image.transpose(1, 0, 2)

        return rgb_image

    # Prepare destination data for NN     prepare_dest_dataset ruft export_to_jpg auf
    def prepare_dest_dataset(self):
        # csv_
        # num | azimuth | categ | dest_filename      | elevation | inst_numb | lighting| mastfilepath         |orienta| set
        # 0   | 4       | animal| animal_000000_08_6_4_4.jgp|6   | 8         | 4       |../norb_master_dir/...|left   | train 
        # csv_prepared_set:  '../norb_dataset_csv/norb-dataset-prepared.csv'
        if exists(self.csv_prepared_set):
            self.logger.info('reading from ' + self.csv_prepared_set)
            return pd.read_csv(self.csv_prepared_set), pd.read_csv(self.csv_prepared_dirs)
        
        # create_dest_structure() gives:
        # norb_dataset_prepared_dirs: | dir "../norb_to_process/shifted_00/train" | orientation | set |
        df = self.create_dest_structure().astype(str)

        # holt alle Pfade mit Bildern die oriant left und Set train und valid getrennt
        train_left_00 = df[(df.orientation=='left')&(df.set=='train')&(df.degree=='0')].dir.values[0] #train_left_00 = df[(df.orientation=='left')&(df.set=='train')].dir.values[0]
        valid_left_00 = df[(df.orientation=='left')&(df.set=='valid')&(df.degree=='0')].dir.values[0] #train_right_00 = df[(df.orientation=='right')&(df.set=='train')].dir.values[0]
        train_left_20 = df[(df.orientation=='left')&(df.set=='train')&(df.degree=='20')].dir.values[0] #train_right = df[(df.orientation=='right')&(df.set=='train')].dir.values[0]
        valid_left_20 = df[(df.orientation=='left')&(df.set=='valid')&(df.degree=='20')].dir.values[0] #valid_right = df[(df.orientation=='right')&(df.set=='valid')].dir.values[0]
        train_left_40 = df[(df.orientation=='left')&(df.set=='train')&(df.degree=='40')].dir.values[0]
        valid_left_40 = df[(df.orientation=='left')&(df.set=='valid')&(df.degree=='40')].dir.values[0]
        train_left_60 = df[(df.orientation=='left')&(df.set=='train')&(df.degree=='60')].dir.values[0]
        valid_left_60 = df[(df.orientation=='left')&(df.set=='valid')&(df.degree=='60')].dir.values[0]
        train_left_80 = df[(df.orientation=='left')&(df.set=='train')&(df.degree=='80')].dir.values[0]
        valid_left_80 = df[(df.orientation=='left')&(df.set=='valid')&(df.degree=='80')].dir.values[0]

        master_df = self.export_to_jpg().astype(str)

        data = []

        for i in range(len(master_df.index)):
            # auffächern aller Eigenschaften des Files aus dem Filenamen
            current_master_row = master_df.iloc[[i]]
            
            current_category = current_master_row.category.values[0]
            current_instance_number = current_master_row.instance_number.values[0]
            current_azimuth = current_master_row.azimuth.values[0]
            current_elevation = current_master_row.elevation.values[0]
            current_lightning = current_master_row.lightning.values[0]
            current_dest_filename = current_master_row.dest_filename.values[0]
            current_master_filepath = current_master_row.master_filepath.values[0]
            current_no = current_master_row.no.values[0]
            current_orientation = current_master_row.orientation.values[0]
            current_set = current_master_row.set.values[0] # train or test

            src_file = current_master_filepath
            # block all right files with unknown engle to left camera
            if current_orientation == 'right':
                continue

            self.logger.info('preparing '+current_dest_filename+' '+current_orientation+' '+current_set)

            # DEGREE = 20
            shift = 20
            peer_file_no = self.mappingToDegree(current_azimuth, shift)
            peer_file_row = master_df[(master_df.category == current_category) # direct access
                        & (master_df.instance_number == current_instance_number)
                        & (master_df.azimuth == str(peer_file_no))
                        & (master_df.elevation == current_elevation)
                        & (master_df.lightning == current_lightning)
                        & (master_df.orientation == current_orientation)
                        & (master_df.set == current_set)]
            # safe sourcefile
            peer_filename20 = peer_file_row.dest_filename.values[0]
            peer_src_20 = peer_file_row.master_filepath.values[0]

            # DEGREE = 40
            shift = 40
            peer_file_no = self.mappingToDegree(current_azimuth, shift)
            peer_file_row = master_df[(master_df.category == current_category) # direct access
                        & (master_df.instance_number == current_instance_number)
                        & (master_df.azimuth == str(peer_file_no))
                        & (master_df.elevation == current_elevation)
                        & (master_df.lightning == current_lightning)
                        & (master_df.orientation == current_orientation)
                        & (master_df.set == current_set)]
            # safe sourcefile
            peer_filename40 = peer_file_row.dest_filename.values[0]
            peer_src_40 = peer_file_row.master_filepath.values[0]

            # DEGREE = 60
            shift = 60
            peer_file_no = self.mappingToDegree(current_azimuth, shift)
            peer_file_row = master_df[(master_df.category == current_category) # direct access
                        & (master_df.instance_number == current_instance_number)
                        & (master_df.azimuth == str(peer_file_no))
                        & (master_df.elevation == current_elevation)
                        & (master_df.lightning == current_lightning)
                        & (master_df.orientation == current_orientation)
                        & (master_df.set == current_set)]
            # safe sourcefile
            peer_filename60 = peer_file_row.dest_filename.values[0]
            peer_src_60 = peer_file_row.master_filepath.values[0]

            # DEGREE = 80
            shift = 80
            peer_file_no = self.mappingToDegree(current_azimuth, shift)
            peer_file_row = master_df[(master_df.category == current_category) # direct access
                        & (master_df.instance_number == current_instance_number)
                        & (master_df.azimuth == str(peer_file_no))
                        & (master_df.elevation == current_elevation)
                        & (master_df.lightning == current_lightning)
                        & (master_df.orientation == current_orientation)
                        & (master_df.set == current_set)]
            # safe sourcefile
            peer_filename80 = peer_file_row.dest_filename.values[0]
            peer_src_80 = peer_file_row.master_filepath.values[0]


            # hier wird der Pfad erzeugt aus train left und z.b. animal, plane, ...
            if (current_set == 'train' and current_orientation == 'left'):
                dir_path = join(train_left_00, current_category)
                dir_path_20 = join(train_left_20, current_category)
                dir_path_40 = join(train_left_40, current_category)
                dir_path_60 = join(train_left_60, current_category)
                dir_path_80 = join(train_left_80, current_category)
            #elif (current_set == 'train' and current_orientation == 'right'):
            #    dir_path = join(train_right, current_category)
            #elif (current_set == 'test' and current_orientation == 'left'):
            #    dir_path = join(valid_left, current_category)
            elif(current_set == 'test' and current_orientation == 'left'):
                dir_path = join(valid_left_00, current_category)
                dir_path_20 = join(valid_left_20, current_category)
                dir_path_40 = join(valid_left_40, current_category)
                dir_path_60 = join(valid_left_60, current_category)
                dir_path_80 = join(valid_left_80, current_category)
            
            # Create category-directory if not existing like animal
            # z.b.:  '../norb_to_process/shifted_00/train/airplane'
            if not exists(dir_path): # einschl. category
                makedirs(dir_path)
            if not exists(dir_path_20): # einschl. category
                makedirs(dir_path_20)
            if not exists(dir_path_40): # einschl. category
                makedirs(dir_path_40)
            if not exists(dir_path_60): # 
                makedirs(dir_path_60)
            if not exists(dir_path_80): # einschl. category
                makedirs(dir_path_80)
            # creating destinationfilepath and a link without left/rigth without filename itsself
            # dst_file: '..norb_to_process/shifted_00/train/animal/animal_0000_08_6_4_4_lt.jpg'
            # src_file: '..norb_master_dir/train/animal_0000_08_6_4_4_lt.jpg'
            # dir_path: z.b. valid_left join current_category
            dst_file_00 = join(dir_path, current_dest_filename)
            if not exists(dst_file_00):
                link(src_file, dst_file_00)
            dst_file_20 = join(dir_path_20, current_dest_filename)
            if not exists(dst_file_20):
                link(peer_src_20, dst_file_20)
            dst_file_40 = join(dir_path_40, current_dest_filename)
            if not exists(dst_file_40):
                link(peer_src_40, dst_file_40)
            dst_file_60 = join(dir_path_60, current_dest_filename)
            if not exists(dst_file_60):
                link(peer_src_60, dst_file_60)
            dst_file_80 = join(dir_path_80, current_dest_filename)
            if not exists(dst_file_80):
                link(peer_src_80, dst_file_80)


            data.append({
                'category': current_category,
                'instance_number': current_instance_number,
                'azimuth': current_azimuth,
                'elevation': current_elevation,
                'lightning': current_lightning,
                'no': current_no,
                'set': current_set,
                'orientation': current_orientation,
                'dest_filename': current_dest_filename,
                'master_filepath': current_master_filepath,
                'src_filepath_00': src_file,
                'src_filepath_20': peer_src_20,
                'src_filepath_40': peer_src_40,
                'src_filepath_60': peer_src_60,
                'src_filepath_80': peer_src_80
             })


        cur_df = pd.DataFrame(data)
        cur_df.to_csv(self.csv_prepared_set, index=False)
        return cur_df, df # return log with files (cur_df) and log with dir structure (df)

    # Create destination directory structure
    def create_dest_structure(self):
        if exists(self.csv_prepared_dirs):
            self.logger.info('Reading from ' + self.csv_prepared_dirs)
            return pd.read_csv(self.csv_prepared_dirs)

        data = []

        self.logger.info('Creating destination structure in ' + self.dest_default)
        dir_path = self.dest_default                    # dest_default = '../norb_to_process
        
        # dir_path_right = join(dir_path, 'right')        # '../norb_to_process/left/train/animal' + right
        # dir_path_left = join(dir_path, 'left')          # '../norb_to_process/left/train/animal' + left
        
        #dir_path_right_train = join(dir_path_right, 'train')
        #data.append({'orientation': 'right', 'set': 'train', 'dir': dir_path_right_train })
        #dir_path_right_valid = join(dir_path_right, 'valid')
        #data.append({'orientation': 'right', 'set': 'valid', 'dir': dir_path_right_valid })
        
        # shifted_00
        dir_path_shifted_00 = join(dir_path, 'shifted_00')
        dir_path_left_train_00 = join(dir_path_shifted_00, 'train') #_left, 'train')
        data.append({'degree': '0','orientation': 'left', 'set': 'train', 'dir': dir_path_left_train_00 })
        dir_path_left_valid_00 = join(dir_path_shifted_00, 'valid') #_left, 'valid')
        data.append({'degree': '0','orientation': 'left', 'set': 'valid', 'dir': dir_path_left_valid_00 })
        
        # shifted_20
        dir_path_shifted_20 = join(dir_path, 'shifted_20')
        dir_path_left_train_20 = join(dir_path_shifted_20, 'train') #_left, 'train')
        data.append({'degree': '20','orientation': 'left', 'set': 'train', 'dir': dir_path_left_train_20 })
        dir_path_left_valid_20 = join(dir_path_shifted_20, 'valid') #_left, 'valid')
        data.append({'degree': '20','orientation': 'left', 'set': 'valid', 'dir': dir_path_left_valid_20 })

        # shifted_40
        dir_path_shifted_40 = join(dir_path, 'shifted_40')
        dir_path_left_train_40 = join(dir_path_shifted_40, 'train') #_left, 'train')
        data.append({'degree': '40','orientation': 'left', 'set': 'train', 'dir': dir_path_left_train_40 })
        dir_path_left_valid_40 = join(dir_path_shifted_40, 'valid') #_left, 'valid')
        data.append({'degree': '40','orientation': 'left', 'set': 'valid', 'dir': dir_path_left_valid_40 })

        # shifted_60
        dir_path_shifted_60 = join(dir_path, 'shifted_60')
        dir_path_left_train_60 = join(dir_path_shifted_60, 'train') #_left, 'train')
        data.append({'degree': '60','orientation': 'left', 'set': 'train', 'dir': dir_path_left_train_60 })
        dir_path_left_valid_60 = join(dir_path_shifted_60, 'valid') #_left, 'valid')
        data.append({'degree': '60','orientation': 'left', 'set': 'valid', 'dir': dir_path_left_valid_60 })

        # shifted_80
        dir_path_shifted_80 = join(dir_path, 'shifted_80')
        dir_path_left_train_80 = join(dir_path_shifted_80, 'train') #_left, 'train')
        data.append({'degree': '80','orientation': 'left', 'set': 'train', 'dir': dir_path_left_train_80 })
        dir_path_left_valid_80 = join(dir_path_shifted_80, 'valid') #_left, 'valid')
        data.append({'degree': '80','orientation': 'left', 'set': 'valid', 'dir': dir_path_left_valid_80 })


        
        dirlist = [ #dir_path_right, \#dir_path_right_train, \#dir_path_left_train, \#dir_path_right_valid, \#dir_path_left_valid 
                    dir_path_shifted_00, \
                    dir_path_left_train_00, \
                    dir_path_left_valid_00, \
                    dir_path_shifted_20, \
                    dir_path_left_train_20, \
                    dir_path_left_valid_20, \
                    dir_path_shifted_40, \
                    dir_path_left_train_40, \
                    dir_path_left_valid_40, \
                    dir_path_shifted_60, \
                    dir_path_left_train_60, \
                    dir_path_left_valid_60, \
                    dir_path_shifted_80, \
                    dir_path_left_train_80, \
                    dir_path_left_valid_80]

        if not exists(dir_path):
            makedirs(dir_path)
 
        for _, dir in enumerate(dirlist):
            if not exists(dir):
                makedirs(dir)

        dir_df = pd.DataFrame(data)
        dir_df.to_csv(self.csv_prepared_dirs, index=False)
        return dir_df

    # FileNumbers are the azimuth degree/10 -> 0,2,4,6,...,32,34
    def mappingToDegree(self, fileNumber, shift):       # fileNumber = 32, , shift 40
        degreePerNumber = 10
        aproxFileNumber = int(fileNumber) + int(shift/degreePerNumber)

        if aproxFileNumber >= 36 :              # 36 >= 34+2 
            currentFileNumber= int(aproxFileNumber) - int(36)
        else:
            currentFileNumber= int(aproxFileNumber)
        return int(currentFileNumber)


    def get_process_dirs(self):
        if exists(self.csv_process_dirs):
            self.logger.info('reading from ' + self.csv_process_dirs)
            return pd.read_csv(self.csv_process_dirs)

        images_df, struct_df = self.prepare_dest_dataset()
        self.logger.info('Get process dirs. Creating ' + self.csv_process_dirs)

        data = []

        for i in range(len(struct_df.index)):
            current_row = struct_df.iloc[[i]]

            current_degree = current_row.degree.values[0]
            current_set = current_row.set.values[0]
            if (current_set == 'valid'):
                continue
            train_dir = current_row.dir.values[0]

            valid_dir = struct_df[(struct_df.degree == current_degree) # direct access
                            & (struct_df.set == 'valid')].dir.values[0]

            data.append({'degree': current_degree, \
                        'train_dir': train_dir, \
                        'valid_dir': valid_dir})

        dir_df = pd.DataFrame(data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        dir_df.to_csv(self.csv_process_dirs, index=False)
        return dir_df

    def get_train_image_class_count(self):
        image_df, dir_df = self.prepare_dest_dataset()
        image_count = len(image_df[(image_df.set=='train')&(image_df.orientation=='left')])
        
        class_count = len(image_df.category.unique())
        return image_count, class_count

    def get_valid_image_class_count(self):
        image_df, dir_df = self.prepare_dest_dataset()
        image_count = len(image_df[(image_df.set=='test')&(image_df.orientation=='left')])
        
        class_count = len(image_df.category.unique())
        return image_count, class_count
    
    def group_dataset_by_category_and_instance(self, dataset_split):
        """
        Group small NORB dataset for (category, instance) key
        
        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        groups: list
            List of 25 groups of 972 elements each. All examples of each group are
            from the same category and instance
        """
        if dataset_split not in ['train', 'test']:
            raise ValueError('Dataset split "{}" not allowed.'.format(dataset_split))

        groups = []
        for key, group in groupby(iterable=sorted(self.data[dataset_split]),
                                  key=lambda x: (x.category, x.instance)):
            groups.append(list(group))

        return groups

    def _fill_data_structures(self, dataset_split):
        """
        Fill SmallNORBDataset data structures for a certain `dataset_split`.
        
        This means all images, category and additional information are loaded from binary
        files of the current split.
        
        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None

        """
        dat_data  = self._parse_NORB_dat_file(self.dataset_files[dataset_split]['dat'])
        cat_data  = self._parse_NORB_cat_file(self.dataset_files[dataset_split]['cat'])
        info_data = self._parse_NORB_info_file(self.dataset_files[dataset_split]['info'])
        for i, small_norb_example in enumerate(self.data[dataset_split]):
            small_norb_example.image_lt   = dat_data[2 * i]
            small_norb_example.image_rt   = dat_data[2 * i + 1]
            small_norb_example.category  = cat_data[i]
            small_norb_example.instance  = info_data[i][0]
            small_norb_example.elevation = info_data[i][1]
            small_norb_example.azimuth   = info_data[i][2]
            small_norb_example.lighting  = info_data[i][3]

    @staticmethod
    def matrix_type_from_magic(magic_number):
        """
        Get matrix data type from magic number
        See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.

        Parameters
        ----------
        magic_number: tuple
            First 4 bytes read from small NORB files 

        Returns
        -------
        element type of the matrix
        """
        convention = {'1E3D4C51': 'single precision matrix',
                      '1E3D4C52': 'packed matrix',
                      '1E3D4C53': 'double precision matrix',
                      '1E3D4C54': 'integer matrix',
                      '1E3D4C55': 'byte matrix',
                      '1E3D4C56': 'short matrix'}
        magic_str = bytearray(reversed(magic_number)).hex().upper()
        return convention[magic_str]

    @staticmethod
    def _parse_small_NORB_header(file_pointer):
        """
        Parse header of small NORB binary file
        
        Parameters
        ----------
        file_pointer: BufferedReader
            File pointer just opened in a small NORB binary file

        Returns
        -------
        file_header_data: dict
            Dictionary containing header information
        """
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': SmallNORBDataset.matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def _parse_NORB_cat_file(file_path):
        """
        Parse small NORB category file
        
        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-cat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,) containing the category of each example
        """
        with open(file_path, mode='rb') as f:
            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, = header['dimensions']

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            examples = np.zeros(shape=num_examples, dtype=np.int32)
            for i in tqdm(range(num_examples), desc='Loading categories...'):
                category, = struct.unpack('<i', f.read(4))
                examples[i] = category

            return examples

    @staticmethod
    def _parse_NORB_dat_file(file_path):
        """
        Parse small NORB data file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-dat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (48600, 96, 96) containing images couples. Each image couple
            is stored in position [i, :, :] and [i+1, :, :]
        """
        with open(file_path, mode='rb') as f:

            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, channels, height, width = header['dimensions']

            examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

            for i in tqdm(range(num_examples * channels), desc='Loading images...'):

                # Read raw image data and restore shape as appropriate
                image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                image = np.uint8(np.reshape(image, newshape=(height, width)))

                examples[i] = image

        return examples

    @staticmethod
    def _parse_NORB_info_file(file_path):
        """
        Parse small NORB information file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-info.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,4) containing the additional info of each example.
            
             - column 1: the instance in the category (0 to 9)
             - column 2: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 
               degrees from the horizontal respectively)
             - column 3: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
             - column 4: the lighting condition (0 to 5)
        """
        with open(file_path, mode='rb') as f:

            header = SmallNORBDataset._parse_small_NORB_header(f)

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            num_examples, num_info = header['dimensions']

            examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)

            for r in tqdm(range(num_examples), desc='Loading info...'):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    examples[r, c] = info

        return examples
