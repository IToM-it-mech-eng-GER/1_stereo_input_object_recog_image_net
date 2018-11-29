import os
import numpy as np
import cv2
import imageio
import argparse
import shutil
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from itertools import product
import scipy.stats
from scipy.sparse import coo_matrix, dia_matrix
from scipy.sparse import linalg
from matplotlib import cm


class WashingtonRGBD(object):
    """

    Data Wrapper class for WashingtonRGBD dataset
    Attributes
    -----------
    root_dir: root directory until the rgbd-dataset folder. For example: ../rgbd-dataset
    csv_dataset: the default directory for loading/saving the csv description of the dataset

    """

    def __init__(self, root_dir='', \
                    csv_dataset='', \
                    csv_aggregated_dataset='', \
                    csv_master_dataset='', \
                    csv_tt_split_01='', \
                    csv_prepared_set='', \
                    csv_prepared_dirs='', \
                    csv_process_dirs='', \
                    master_default='', \
                    dest_default='', \
                    dest_size=224, \
                    split_rate=0.2):
        self.logger = logging.getLogger(__name__)
        self.root_dir = root_dir
        self.csv_dataset = csv_dataset
        self.csv_aggregated_dataset = csv_aggregated_dataset
        self.csv_master_dataset = csv_master_dataset
        self.csv_tt_split_01 = csv_tt_split_01
        self.csv_prepared_set = csv_prepared_set
        self.csv_prepared_dirs = csv_prepared_dirs
        self.csv_process_dirs = csv_process_dirs
        self.master_default = master_default
        self.dest_default = dest_default
        self.dest_size = dest_size
        self.split_rate = split_rate
        self.master_dir_filled = os.path.join(master_default, 'master_dir_filled')
        self.master_dir_notfilled = os.path.join(master_default, 'master_dir_notfilled')

        if not os.path.isdir(self.master_dir_filled):
            os.mkdir(self.master_dir_filled)
        if not os.path.isdir(self.master_dir_notfilled):
            os.mkdir(self.master_dir_notfilled)


    # Load the dataset metadata to a Pandas dataframe and save the result to a csv file
    # if it does not exists
    # otherwise read the csv
    # The missing pose values will be saved as -1
    def load_metadata(self):
        if os.path.isfile(self.csv_dataset):
            self.logger.info('reading from ' + self.csv_dataset)
            return pd.read_csv(self.csv_dataset)

        file_list = os.walk(self.root_dir)

        data = []

        for current_root, _, files in file_list:
 
            for f in files:

                name_components = f.split('_')

                # The category name can be 1 word or 2 words, such as 'apple' or 'cell_phone'
                # So, when splitting the file name by '_', there can be 5 or 6 components
                # That's why I read the name backward to make sure I get the proper data pieces
                # reversed_name_components = np.flip(name_components, axis=0)

                if len(name_components) < 5:
                    continue

                n_components = len(name_components)
                if n_components > 5:    # if n_components > 5, it means the category name has more than 1 word
                    category = '_'.join(name_components[0: n_components - 4])
                else:
                    category = name_components[0]

                instance_number = name_components[-4]
                video_no = name_components[-3]
                frame_no = name_components[-2]
                data_type = name_components[-1].split('.')[0]
                if (data_type == 'loc' or data_type == 'maskcrop'): # not necessary
                    continue

                self.logger.info("processing " + f)

                data.append({'location': os.path.join(current_root, f),
                             'category': category,
                             'instance_number': int(instance_number),
                             'video_no': int(video_no),
                             'frame_no': int(frame_no),
                             'data_type': data_type})

        data_frame = pd.DataFrame(data) \
            .sort_values(['category', 'instance_number', 'video_no', 'frame_no'])

        self.logger.info("csv saved to file: " + self.csv_dataset)
        data_frame.to_csv(self.csv_dataset, index=False)

        return data_frame

  
    # Get a new dataframe where each row represent all information about 1 frame including the rgb and depth locations
    # structure: ['category', 'instance_number', 'video_no', 'frame_no', 'crop_location', 'depthcrop_location', 
    # 'dest_filename']         
    def aggregate_frame_data(self):
        if os.path.isfile(self.csv_aggregated_dataset):
            self.logger.info('reading from ' + self.csv_aggregated_dataset)
            return pd.read_csv(self.csv_aggregated_dataset)

        raw_df = self.load_metadata()

        raw_rgb_df = raw_df[raw_df.data_type == 'crop']
        raw_depth_df = raw_df[raw_df.data_type == 'depthcrop']
        
        data = []

        for i in range(len(raw_rgb_df.index)):
            current_rgb_row = raw_rgb_df.iloc[[i]]

            current_category = current_rgb_row.category.values[0]
            current_instance_number = current_rgb_row.instance_number.values[0]
            current_video_no = current_rgb_row.video_no.values[0]
            current_frame_no = current_rgb_row.frame_no.values[0]
            dest_filename = current_category \
                            + '_' + str(current_instance_number) \
                            + '_' + str(current_video_no) \
                            + '_' + str(current_frame_no) \
                            + '.png'
            current_crop_location = current_rgb_row.location.values[0]

            current_depthcrop_location = raw_depth_df[(raw_depth_df.category == current_category)
                                                      & (raw_depth_df.instance_number == current_instance_number)
                                                      & (raw_depth_df.video_no == current_video_no)
                                                      & (raw_depth_df.frame_no == current_frame_no)].location.values[0]

            self.logger.info("processing " + os.path.split(current_crop_location)[1]
                             + " and " + os.path.split(current_depthcrop_location)[1])

            data.append({
                'category': current_category,
                'instance_number': current_instance_number,
                'video_no': current_video_no,
                'frame_no': current_frame_no,
                'crop_location': current_crop_location,
                'depthcrop_location': current_depthcrop_location,
                'dest_filename': dest_filename
             })

        new_df = pd.DataFrame(data)
        new_df.to_csv(self.csv_aggregated_dataset, index=False)
        return new_df

    # Tile the image to a destination square image, used in Eitel et al.
    #
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

    def fill_depth_colorization(self, imgRgb, imgDepth, alpha=1.0):
        """
        Preprocesses the kinect depth image using a gray scale version of the
        RGB image as a weighting for the smoothing. This code is a slight
        adaptation of Anat Levin's colorization code:

        See: www.cs.huji.ac.il/~yweiss/Colorization/

        Args:
        imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
                be between 0 and 1.
        imgDepth - HxW matrix, the depth image for the current frame in
                    absolute (meters) space.
        alpha - a penalty value between 0 and 1 for the current depth values.
        Port to Python based on https://gist.github.com/bwaldvogel/6892721
        """

        assert np.isnan(imgDepth).sum() == 0 # Fehlstellen sind mit validen Zahlenwerten gefüllt 
        
        imgIsNoise=imgDepth<11 # Maske mit False als Inhalt bei Wert<11 (Fehlstelle), Annahme Silberman bis 10-> Fehlstelle
        
        minImgAbsDepth=np.min(imgDepth, axis=None) #TM bei 0 - Fehlstelle
        maxImgAbsDepth=np.max(imgDepth, axis=None) #TM Pixelhöchstwert
        meanImgDepth = np.mean(imgDepth, axis=None)
        
        imgDepth -= minImgAbsDepth # Werte können anscheinend bis 10 reichen, dann auf Px-minImgAbsDepth normalisieren
        maxNormImgAbsDepth = np.max(imgDepth, axis=None)
        imgDepth = imgDepth / maxNormImgAbsDepth

        knownValMask = imgIsNoise==False #TM Bekannte Werte in Maske mit True belegen, wenn dort kein imageNoise vorhanden ist
        imgDepth[~knownValMask] = 0 # alle Fehlstellen mit 0 belegen

        H, W = imgDepth.shape
        numPix = H * W

        indsM = np.arange(numPix).reshape(H, W)

        # convert to gray image
        #grayImg = imgRgb.mean(axis=2)
        
        grayImg = cv2.cvtColor(imgRgb, cv2.COLOR_RGB2GRAY) # Liste von Ints als Ergebnis
        maxPix = np.max(grayImg, axis=None)
        grayImg = grayImg / maxPix # float im Wertebereich 0.0 bis 1.0


        winRad = 1  # Fenstergröße

        tlen = 0
        absImgNdx = 0
        winPixel = (2 * winRad + 1) ** 2
        cols = np.zeros(numPix * winPixel)
        rows = np.zeros(numPix * winPixel)
        vals = np.zeros(numPix * winPixel)
        gvals = np.zeros(winPixel)

        for absImgNdx, (i, j) in enumerate(product(range(H), range(W))):

            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[tlen] = absImgNdx
                    cols[tlen] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    tlen += 1
                    nWin += 1

            #if j == 0 and i % 10 == 0:
            #    print(i, j)

            curVal = grayImg[i, j]  #TMOK
            gvals[nWin] = curVal    #TMOK

            silberman = True
            # FROM HERE
            if silberman==False:
                assert np.sum(gvals[:nWin]) > 0, "gvals: %s" % (repr(gvals[:nWin]))
                c_var = np.var(gvals[:nWin])
                # c_var = np.mean((gvals(1:nWin+1)-mean(gvals(1:nWin+1))).^2)

                csig = c_var
                # csig = c_var * 0.6
                # TODO
                # mgv = min((gvals(1:nWin)-curVal).^2)
                # if csig < (-mgv/log(0.01)):
                #     csig=-mgv/log(0.01)

                csig = max(csig, 0.000002)
            # TO HERE
            
            # FROM SILBERMAN
            if silberman==True:
                c_var=np.mean((gvals[:nWin-1] - np.mean(gvals[:nWin-1])) ** 2)
                csig = c_var * 0.6 
                na = (gvals[:nWin-1]-curVal)**2
                try:
                    mgv = np.min(na)
                except:
                    print(str(absImgNdx) + ' i'+str(i)+' j'+str(j)+'  ii'+str(ii)+' jj'+str(jj))
                    
                if csig < (- mgv / np.log(0.01)):
                    csig=- mgv / np.log(0.01)
                csig = max(csig, 0.000002)    #TM
            # TO SILBERMAN

            # gvals(1:nWin) = exp(-(gvals(1:nWin)-curVal).^2/csig)
            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            # gvals(1:nWin) = gvals(1:nWin) / sum(gvals(1:nWin))
            s = gvals[:nWin].sum()
            if s > 0:
                gvals[:nWin] /= s

                s = np.round(gvals[:nWin].sum(), 6)
                assert s == 1, "expected sum to be 1: %.10f" % s

            # vals(len-nWin+1 : len) = -gvals(1:nWin)
            vals[tlen - nWin:tlen] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[tlen] = absImgNdx
            cols[tlen] = absImgNdx
            assert vals[tlen] == 0     # not yet set
            vals[tlen] = 1             # sum(gvals(1:nWin))

            tlen += 1

        assert tlen <= numPix * winPixel, "%d > %d" % (tlen, numPix * winPixel)

        A = coo_matrix((vals, (rows, cols)), shape=(numPix, numPix))

        vals = knownValMask.flatten() * alpha

        G = dia_matrix((vals, 0), shape=(numPix, numPix))

        new_vals = linalg.spsolve((A + G), vals * imgDepth.flatten())
        new_vals.shape = H, W

        # denormalize
        new_vals *= maxNormImgAbsDepth
        new_vals += minImgAbsDepth

        return new_vals.astype(np.uint16)


    # grey-rescaling and colorizing the depth map using jet color map, no interpolation of missing values
    def colorize_depth(self, depth_map):
        # scale everything to [0, 255]
        sorted_depth = np.unique(np.sort(depth_map.flatten()))
        min_depth = sorted_depth[1] # bei [0] stehen fehlende Werte, nur zwischen existierenden Werten normalisieren
        max_depth = sorted_depth[len(sorted_depth) - 1]
        depth_map = np.asarray([abs(pixel - min_depth) * 1.0 / (max_depth - min_depth) for pixel in depth_map])
        depth_map = np.uint8(cm.jet_r(depth_map) * 255)
        return depth_map[:, :, 0:3]

    # grey-scaling and colorizing, but use interpolated with Nathan Silberman's code missing values picture 
    def colorize_depth_flattened_picture(self, depth_map):
        # scale everything to [0, 255]
        sorted_depth = np.unique(np.sort(depth_map.flatten()))
        min_depth = sorted_depth[0] 
        max_depth = sorted_depth[len(sorted_depth) - 1]
        depth_map = np.asarray([(pixel - min_depth) * 1.0 / (max_depth - min_depth) for pixel in depth_map])
        depth_map = np.uint8(cm.jet_r(depth_map) * 255)
        return depth_map[:, :, 0:3]
     
    # Fill missing depth values, colorize the image and save into a destination file
    def preprocess_frame(self, crop, depthcrop, dest_file, fill=True):
        if os.path.isfile(dest_file):
            self.logger.info('file ' + dest_file + ' already exists. Continuing...')
            return
        try:
            rgb_img = imageio.imread(crop)
            depth_img = imageio.imread(depthcrop)

            if (fill==True):
                #fill missing values
                filled_img = self.fill_depth_colorization(rgb_img, depth_img)
                #tile borders
                rgb_img = self.tile_border(rgb_img)
                target_img = self.tile_border(self.colorize_depth_flattened_picture(filled_img))
            else:
                #tile borders
                rgb_img = self.tile_border(rgb_img)
                target_img = self.tile_border(self.colorize_depth(depth_img))


            combined_image = np.concatenate([rgb_img, target_img], axis=1)
            cv2.imwrite(dest_file, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

        except IOError:
            self.logger.info('file ' + crop + ' does not exist')
    
    # Split the dataset in training and validation part
    # Version 01: use scipy train_test_split() 
    def train_test_split_01(self):
        self.logger.info('Splitting train/test with rate=' + str(self.split_rate))
        split_rate = self.split_rate
        if os.path.isfile(self.csv_tt_split_01):
            self.logger.info('reading from ' + self.csv_tt_split_01)
            return pd.read_csv(self.csv_tt_split_01)

        washington_master_df = self.create_rgbd_images()

        # initializing data frame structure
        train_df_all = pd.DataFrame(columns=list(washington_master_df))
        valid_df_all = pd.DataFrame(columns=list(washington_master_df))

        categories = washington_master_df.category.unique()

        for cat in categories:
            category_df = washington_master_df[washington_master_df.category == cat]
            # split separately for each category 
            train_df, valid_df = train_test_split(category_df, test_size=split_rate)

            train_df_all = train_df_all.append(train_df)
            valid_df_all = valid_df_all.append(valid_df)

        train_df_all['set'] = 'train'
        valid_df_all['set'] = 'valid'
        next_df = pd.concat([train_df_all, valid_df_all])
        next_df = next_df.sort_values(['category', 'instance_number', 'video_no', 'frame_no'])
        next_df.to_csv(self.csv_tt_split_01, index=False)
        return next_df

    def mappingToDegree(self, fileNumber, totalNum, shift):
        degreePerNumber = 360/totalNum           # = 1,70616
        aproxFileNumber = shift/degreePerNumber   # = 5,2750
        aproxFileNumber = int(round(aproxFileNumber, 0))

        if fileNumber <= totalNum - aproxFileNumber:              # curr <= 211-5 
            currentFileNumber= int(aproxFileNumber) + int(fileNumber)
        else:                                                     # curr <= 211-5 total-curr
            currentFileNumber= -1*(totalNum - aproxFileNumber - fileNumber)
        
        return int(currentFileNumber)

    # Prepare destination data for NN
    def prepare_dest_dataset(self):
        
        if os.path.isfile(self.csv_prepared_set):
            self.logger.info('reading from ' + self.csv_prepared_set)
            return pd.read_csv(self.csv_prepared_set), pd.read_csv(self.csv_prepared_dirs)
        
        df = self.create_dest_structure().astype(str)

        shifted_00_filled_train = df[(df.degree=='0')&(df.filled_set=='filled')&(df.dataset=='train')].dir.values[0]
        shifted_00_filled_valid = df[(df.degree=='0')&(df.filled_set=='filled')&(df.dataset=='valid')].dir.values[0]
        shifted_00_notfilled_train = df[(df.degree=='0')&(df.filled_set=='unfilled')&(df.dataset=='train')].dir.values[0]
        shifted_00_notfilled_valid = df[(df.degree=='0')&(df.filled_set=='unfilled')&(df.dataset=='valid')].dir.values[0]
        shifted_20_filled_train = df[(df.degree=='20')&(df.filled_set=='filled')&(df.dataset=='train')].dir.values[0]
        shifted_20_filled_valid = df[(df.degree=='20')&(df.filled_set=='filled')&(df.dataset=='valid')].dir.values[0]
        shifted_20_notfilled_train = df[(df.degree=='20')&(df.filled_set=='unfilled')&(df.dataset=='train')].dir.values[0]
        shifted_20_notfilled_valid = df[(df.degree=='20')&(df.filled_set=='unfilled')&(df.dataset=='valid')].dir.values[0]
        shifted_40_filled_train = df[(df.degree=='40')&(df.filled_set=='filled')&(df.dataset=='train')].dir.values[0]
        shifted_40_filled_valid = df[(df.degree=='40')&(df.filled_set=='filled')&(df.dataset=='valid')].dir.values[0]
        shifted_40_notfilled_train = df[(df.degree=='40')&(df.filled_set=='unfilled')&(df.dataset=='train')].dir.values[0]
        shifted_40_notfilled_valid = df[(df.degree=='40')&(df.filled_set=='unfilled')&(df.dataset=='valid')].dir.values[0]
        shifted_60_filled_train = df[(df.degree=='60')&(df.filled_set=='filled')&(df.dataset=='train')].dir.values[0]
        shifted_60_filled_valid = df[(df.degree=='60')&(df.filled_set=='filled')&(df.dataset=='valid')].dir.values[0]
        shifted_60_notfilled_train = df[(df.degree=='60')&(df.filled_set=='unfilled')&(df.dataset=='train')].dir.values[0]
        shifted_60_notfilled_valid = df[(df.degree=='60')&(df.filled_set=='unfilled')&(df.dataset=='valid')].dir.values[0]
        shifted_80_filled_train = df[(df.degree=='80')&(df.filled_set=='filled')&(df.dataset=='train')].dir.values[0]
        shifted_80_filled_valid = df[(df.degree=='80')&(df.filled_set=='filled')&(df.dataset=='valid')].dir.values[0]
        shifted_80_notfilled_train = df[(df.degree=='80')&(df.filled_set=='unfilled')&(df.dataset=='train')].dir.values[0]
        shifted_80_notfilled_valid = df[(df.degree=='80')&(df.filled_set=='unfilled')&(df.dataset=='valid')].dir.values[0]       
        
        tt_split_df = self.train_test_split_01()

        data = []

        for i in range(len(tt_split_df.index)):
            current_master_row = tt_split_df.iloc[[i]]

            current_category = current_master_row.category.values[0]
            current_instance_number = current_master_row.instance_number.values[0]
            current_video_no = current_master_row.video_no.values[0]
            current_frame_no = current_master_row.frame_no.values[0]
            current_filename = current_master_row.dest_filename.values[0]
            current_dest_filename_filled = current_master_row.master_filepath_filled.values[0]
            current_dest_filename_notfilled = current_master_row.master_filepath_notfilled.values[0]
            current_crop_location = current_master_row.crop_location.values[0]
            current_depthcrop_location = current_master_row.depthcrop_location.values[0]
            current_set = current_master_row.set.values[0] # train or test

            frame_no_max = tt_split_df[(tt_split_df.category == current_category)
                        & (tt_split_df.instance_number == current_instance_number)
                        & (tt_split_df.video_no == current_video_no)].frame_no.max()
            
            tol = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10]
            # tolerance +/-10 for missing images (e.g. dry_battery 2_1_106-116)

            # DEGREE = 20
            DEGREE = 20
            for i in range(len(tol)):
                try:
                    peer_file_no = self.mappingToDegree(current_frame_no + tol[i], frame_no_max, DEGREE)
                    peer_file_row = tt_split_df[(tt_split_df.category == current_category) # direct access
                                & (tt_split_df.instance_number == current_instance_number)
                                & (tt_split_df.video_no == current_video_no)
                                & (tt_split_df.frame_no == peer_file_no)]
                
                        
                    peer_filename20 = peer_file_row.dest_filename.values[0]
                    break
                except IndexError:
                    self.logger.warning('File ' + current_category + ' ' + str(current_instance_number) + ' ' \
                            + str(current_video_no) + ' ' + str(peer_file_no) + ' not found')
                    continue
            
            peer_src_filled20 = peer_file_row.master_filepath_filled.values[0]
            peer_src_notfilled20 = peer_file_row.master_filepath_notfilled.values[0]

            # DEGREE = 40
            DEGREE = 40
            for i in range(len(tol)):
                try:
                    peer_file_no = self.mappingToDegree(current_frame_no + tol[i], frame_no_max, DEGREE)
                    peer_file_row = tt_split_df[(tt_split_df.category == current_category) # direct access
                                & (tt_split_df.instance_number == current_instance_number)
                                & (tt_split_df.video_no == current_video_no)
                                & (tt_split_df.frame_no == peer_file_no)]
                
                        
                    peer_filename40 = peer_file_row.dest_filename.values[0]
                    break
                except IndexError:
                    self.logger.warning('File ' + current_category + ' ' + str(current_instance_number) + ' ' \
                            + str(current_video_no) + ' ' + str(peer_file_no) + ' not found')
                    continue
            
            peer_src_filled40 = peer_file_row.master_filepath_filled.values[0]
            peer_src_notfilled40 = peer_file_row.master_filepath_notfilled.values[0]

            # DEGREE = 60
            DEGREE = 60
            for i in range(len(tol)):
                try:
                    peer_file_no = self.mappingToDegree(current_frame_no + tol[i], frame_no_max, DEGREE)
                    peer_file_row = tt_split_df[(tt_split_df.category == current_category) # direct access
                                & (tt_split_df.instance_number == current_instance_number)
                                & (tt_split_df.video_no == current_video_no)
                                & (tt_split_df.frame_no == peer_file_no)]
                
                        
                    peer_filename60 = peer_file_row.dest_filename.values[0]
                    break
                except IndexError:
                    self.logger.warning('File ' + current_category + ' ' + str(current_instance_number) + ' ' \
                            + str(current_video_no) + ' ' + str(peer_file_no) + ' not found')
                    continue
            
            peer_src_filled60 = peer_file_row.master_filepath_filled.values[0]
            peer_src_notfilled60 = peer_file_row.master_filepath_notfilled.values[0]

            # DEGREE = 80
            DEGREE = 80
            for i in range(len(tol)):
                try:
                    peer_file_no = self.mappingToDegree(current_frame_no + tol[i], frame_no_max, DEGREE)
                    peer_file_row = tt_split_df[(tt_split_df.category == current_category) # direct access
                                & (tt_split_df.instance_number == current_instance_number)
                                & (tt_split_df.video_no == current_video_no)
                                & (tt_split_df.frame_no == peer_file_no)]
                
                        
                    peer_filename80 = peer_file_row.dest_filename.values[0]
                    break
                except IndexError:
                    self.logger.warning('File ' + current_category + ' ' + str(current_instance_number) + ' ' \
                            + str(current_video_no) + ' ' + str(peer_file_no) + ' not found')
                    continue
            
            peer_src_filled80 = peer_file_row.master_filepath_filled.values[0]
            peer_src_notfilled80 = peer_file_row.master_filepath_notfilled.values[0]


            src_filled = current_dest_filename_filled
            src_notfilled = current_dest_filename_notfilled

            self.logger.info('preparing ' + current_filename + ' 20: ' + peer_filename20 + ' 40: ' + peer_filename40 \
                            + ' 60: ' + peer_filename60 + ' 80: ' + peer_filename80)

            # process image
            if (current_set == 'train'):
                # filled
                dir_path = os.path.join(shifted_00_filled_train, current_category)
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                dst_filled = os.path.join(dir_path, current_filename)
                if not os.path.isfile(dst_filled):
                    os.link(src_filled, dst_filled)
                # 20 filled
                peer_dir_path20 = os.path.join(shifted_20_filled_train, current_category)
                if not os.path.isdir(peer_dir_path20):
                    os.mkdir(peer_dir_path20)
                peer_dst_filled20 = os.path.join(peer_dir_path20, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled20):
                    os.link(peer_src_filled20, peer_dst_filled20)
                # 40 filled
                peer_dir_path40 = os.path.join(shifted_40_filled_train, current_category)
                if not os.path.isdir(peer_dir_path40):
                    os.mkdir(peer_dir_path40)
                peer_dst_filled40 = os.path.join(peer_dir_path40, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled40):
                    os.link(peer_src_filled40, peer_dst_filled40)
                # 60 filled
                peer_dir_path60 = os.path.join(shifted_60_filled_train, current_category)
                if not os.path.isdir(peer_dir_path60):
                    os.mkdir(peer_dir_path60)
                peer_dst_filled60 = os.path.join(peer_dir_path60, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled60):
                    os.link(peer_src_filled60, peer_dst_filled60)
                # 80 filled
                peer_dir_path80 = os.path.join(shifted_80_filled_train, current_category)
                if not os.path.isdir(peer_dir_path80):
                    os.mkdir(peer_dir_path80)
                peer_dst_filled80 = os.path.join(peer_dir_path80, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled80):
                    os.link(peer_src_filled80, peer_dst_filled80)

                # not filled 
                dir_path = os.path.join(shifted_00_notfilled_train, current_category)
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                dst_notfilled = os.path.join(dir_path, current_filename)
                if not os.path.isfile(dst_notfilled):
                    os.link(src_notfilled, dst_notfilled)
                # 20 notfilled
                peer_dir_path20 = os.path.join(shifted_20_notfilled_train, current_category)
                if not os.path.isdir(peer_dir_path20):
                    os.mkdir(peer_dir_path20)
                peer_dst_notfilled20 = os.path.join(peer_dir_path20, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled20):
                    os.link(peer_src_notfilled20, peer_dst_notfilled20)
                # 40 notfilled
                peer_dir_path40 = os.path.join(shifted_40_notfilled_train, current_category)
                if not os.path.isdir(peer_dir_path40):
                    os.mkdir(peer_dir_path40)
                peer_dst_notfilled40 = os.path.join(peer_dir_path40, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled40):
                    os.link(peer_src_notfilled40, peer_dst_notfilled40)
                # 60 notfilled
                peer_dir_path60 = os.path.join(shifted_60_notfilled_train, current_category)
                if not os.path.isdir(peer_dir_path60):
                    os.mkdir(peer_dir_path60)
                peer_dst_notfilled60 = os.path.join(peer_dir_path60, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled60):
                    os.link(peer_src_notfilled60, peer_dst_notfilled60)
                # 80 notfilled
                peer_dir_path80 = os.path.join(shifted_80_notfilled_train, current_category)
                if not os.path.isdir(peer_dir_path80):
                    os.mkdir(peer_dir_path80)
                peer_dst_notfilled80 = os.path.join(peer_dir_path80, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled80):
                    os.link(peer_src_notfilled80, peer_dst_notfilled80)

            else: # valid
                # filled
                dir_path = os.path.join(shifted_00_filled_valid, current_category)
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                dst_filled = os.path.join(dir_path, current_filename)
                if not os.path.isfile(dst_filled):
                    os.link(src_filled, dst_filled)
                # 20 filled
                peer_dir_path20 = os.path.join(shifted_20_filled_valid, current_category)
                if not os.path.isdir(peer_dir_path20):
                    os.mkdir(peer_dir_path20)
                peer_dst_filled20 = os.path.join(peer_dir_path20, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled20):
                    os.link(peer_src_filled20, peer_dst_filled20)
                # 40 filled
                peer_dir_path40 = os.path.join(shifted_40_filled_valid, current_category)
                if not os.path.isdir(peer_dir_path40):
                    os.mkdir(peer_dir_path40)
                peer_dst_filled40 = os.path.join(peer_dir_path40, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled40):
                    os.link(peer_src_filled40, peer_dst_filled40)
                # 60 filled
                peer_dir_path60 = os.path.join(shifted_60_filled_valid, current_category)
                if not os.path.isdir(peer_dir_path60):
                    os.mkdir(peer_dir_path60)
                peer_dst_filled60 = os.path.join(peer_dir_path60, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled60):
                    os.link(peer_src_filled60, peer_dst_filled60)
                # 80 filled
                peer_dir_path80 = os.path.join(shifted_80_filled_valid, current_category)
                if not os.path.isdir(peer_dir_path80):
                    os.mkdir(peer_dir_path80)
                peer_dst_filled80 = os.path.join(peer_dir_path80, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_filled80):
                    os.link(peer_src_filled80, peer_dst_filled80)

                # not filled 
                dir_path = os.path.join(shifted_00_notfilled_valid, current_category)
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                dst_notfilled = os.path.join(dir_path, current_filename)
                if not os.path.isfile(dst_notfilled):
                    os.link(src_notfilled, dst_notfilled)
                # 20 notfilled
                peer_dir_path20 = os.path.join(shifted_20_notfilled_valid, current_category)
                if not os.path.isdir(peer_dir_path20):
                    os.mkdir(peer_dir_path20)
                peer_dst_notfilled20 = os.path.join(peer_dir_path20, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled20):
                    os.link(peer_src_notfilled20, peer_dst_notfilled20)
                # 40 notfilled
                peer_dir_path40 = os.path.join(shifted_40_notfilled_valid, current_category)
                if not os.path.isdir(peer_dir_path40):
                    os.mkdir(peer_dir_path40)
                peer_dst_notfilled40 = os.path.join(peer_dir_path40, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled40):
                    os.link(peer_src_notfilled40, peer_dst_notfilled40)
                # 60 notfilled
                peer_dir_path60 = os.path.join(shifted_60_notfilled_valid, current_category)
                if not os.path.isdir(peer_dir_path60):
                    os.mkdir(peer_dir_path60)
                peer_dst_notfilled60 = os.path.join(peer_dir_path60, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled60):
                    os.link(peer_src_notfilled60, peer_dst_notfilled60)
                # 80 notfilled
                peer_dir_path80 = os.path.join(shifted_80_notfilled_valid, current_category)
                if not os.path.isdir(peer_dir_path80):
                    os.mkdir(peer_dir_path80)
                peer_dst_notfilled80 = os.path.join(peer_dir_path80, current_filename) # unter dem gleichen Filenamen ablegen
                if not os.path.isfile(peer_dst_notfilled80):
                    os.link(peer_src_notfilled80, peer_dst_notfilled80)

 
            data.append({
                'category': current_category,
                'instance_number': current_instance_number,
                'video_no': current_video_no,
                'frame_no': current_frame_no,
                'crop_location': current_crop_location,
                'depthcrop_location': current_depthcrop_location,
                'dest_filename': current_filename,
                'master_filepath_filled': current_dest_filename_filled,
                'master_filepath_notfilled': current_dest_filename_notfilled,
                'prepared_filepath_filled': dst_filled,
                'prepared_filepath_notfilled': dst_notfilled,
                'prepared_20degree_filled':  peer_src_filled20, # 20
                'prepared_20degree_notfilled': peer_src_notfilled20, # 20
                'prepared_40degree_filled':  peer_src_filled40, # 40
                'prepared_40degree_notfilled': peer_src_notfilled40, # 40
                'prepared_60degree_filled':  peer_src_filled60, # 60
                'prepared_60degree_notfilled': peer_src_notfilled60, # 60
                'prepared_80degree_filled':  peer_src_filled80, # 80
                'prepared_80degree_notfilled': peer_src_notfilled80, # 80
                'set': current_set
             })

        cur_df = pd.DataFrame(data)
        cur_df.to_csv(self.csv_prepared_set, index=False)
        return cur_df, df # return log with files (cur_df) and log with dir structure (df)

    # Create destination directory structure
    def create_dest_structure(self):
        if os.path.isfile(self.csv_prepared_dirs):
            self.logger.info('Reading from ' + self.csv_prepared_dirs)
            return pd.read_csv(self.csv_prepared_dirs)

        data = []

        self.logger.info('Creating destination structure in ' + self.dest_default)
        dir_path = self.dest_default
        dir_path_filled = os.path.join(dir_path, 'images_filled') #self.dest_dir_filled
        dir_path_notfilled = os.path.join(dir_path, 'images_notfilled') #self.dest_dir_notfilled
        
        shifted_00_dir_filled = os.path.join(dir_path_filled, 'shifted_00')
        shifted_00_filled_train = os.path.join(shifted_00_dir_filled, 'train')
        data.append({'degree': '0', 'filled_set': 'filled', 'dataset': 'train', 'dir': shifted_00_filled_train })
        shifted_00_filled_valid = os.path.join(shifted_00_dir_filled, 'valid')
        data.append({'degree': '0', 'filled_set': 'filled', 'dataset': 'valid', 'dir': shifted_00_filled_valid })

        shifted_00_dir_notfilled = os.path.join(dir_path_notfilled, 'shifted_00')
        shifted_00_notfilled_train = os.path.join(shifted_00_dir_notfilled, 'train')
        data.append({'degree': '0', 'filled_set': 'unfilled', 'dataset': 'train', 'dir': shifted_00_notfilled_train })
        shifted_00_notfilled_valid = os.path.join(shifted_00_dir_notfilled, 'valid')
        data.append({'degree': '0', 'filled_set': 'unfilled', 'dataset': 'valid', 'dir': shifted_00_notfilled_valid })
          
        shifted_20_dir_filled = os.path.join(dir_path_filled, 'shifted_20')
        shifted_20_filled_train = os.path.join(shifted_20_dir_filled, 'train')
        data.append({'degree': '20', 'filled_set': 'filled', 'dataset': 'train', 'dir': shifted_20_filled_train })
        shifted_20_filled_valid = os.path.join(shifted_20_dir_filled, 'valid')
        data.append({'degree': '20', 'filled_set': 'filled', 'dataset': 'valid', 'dir': shifted_20_filled_valid })
        
        shifted_20_dir_notfilled = os.path.join(dir_path_notfilled, 'shifted_20')
        shifted_20_notfilled_train = os.path.join(shifted_20_dir_notfilled, 'train')
        data.append({'degree': '20', 'filled_set': 'unfilled', 'dataset': 'train', 'dir': shifted_20_notfilled_train })
        shifted_20_notfilled_valid = os.path.join(shifted_20_dir_notfilled, 'valid')
        data.append({'degree': '20', 'filled_set': 'unfilled', 'dataset': 'valid', 'dir': shifted_20_notfilled_valid })

        shifted_40_dir_filled = os.path.join(dir_path_filled, 'shifted_40')
        shifted_40_filled_train = os.path.join(shifted_40_dir_filled, 'train')
        data.append({'degree': '40', 'filled_set': 'filled', 'dataset': 'train', 'dir': shifted_40_filled_train })
        shifted_40_filled_valid = os.path.join(shifted_40_dir_filled, 'valid')
        data.append({'degree': '40', 'filled_set': 'filled', 'dataset': 'valid', 'dir': shifted_40_filled_valid })
        
        shifted_40_dir_notfilled = os.path.join(dir_path_notfilled, 'shifted_40')
        shifted_40_notfilled_train = os.path.join(shifted_40_dir_notfilled, 'train')
        data.append({'degree': '40', 'filled_set': 'unfilled', 'dataset': 'train', 'dir': shifted_40_notfilled_train })
        shifted_40_notfilled_valid = os.path.join(shifted_40_dir_notfilled, 'valid')
        data.append({'degree': '40', 'filled_set': 'unfilled', 'dataset': 'valid', 'dir': shifted_40_notfilled_valid })
        
        shifted_60_dir_filled = os.path.join(dir_path_filled, 'shifted_60')
        shifted_60_filled_train = os.path.join(shifted_60_dir_filled, 'train')
        data.append({'degree': '60', 'filled_set': 'filled', 'dataset': 'train', 'dir': shifted_60_filled_train })
        shifted_60_filled_valid = os.path.join(shifted_60_dir_filled, 'valid')
        data.append({'degree': '60', 'filled_set': 'filled', 'dataset': 'valid', 'dir': shifted_60_filled_valid })
        
        shifted_60_dir_notfilled = os.path.join(dir_path_notfilled, 'shifted_60')
        shifted_60_notfilled_train = os.path.join(shifted_60_dir_notfilled, 'train')
        data.append({'degree': '60', 'filled_set': 'unfilled', 'dataset': 'train', 'dir': shifted_60_notfilled_train })
        shifted_60_notfilled_valid = os.path.join(shifted_60_dir_notfilled, 'valid')
        data.append({'degree': '60', 'filled_set': 'unfilled', 'dataset': 'valid', 'dir': shifted_60_notfilled_valid })
        
        shifted_80_dir_filled = os.path.join(dir_path_filled, 'shifted_80')
        shifted_80_filled_train = os.path.join(shifted_80_dir_filled, 'train')
        data.append({'degree': '80', 'filled_set': 'filled', 'dataset': 'train', 'dir': shifted_80_filled_train })
        shifted_80_filled_valid = os.path.join(shifted_80_dir_filled, 'valid')
        data.append({'degree': '80', 'filled_set': 'filled', 'dataset': 'valid', 'dir': shifted_80_filled_valid })
        
        shifted_80_dir_notfilled = os.path.join(dir_path_notfilled, 'shifted_80')
        shifted_80_notfilled_train = os.path.join(shifted_80_dir_notfilled, 'train')
        data.append({'degree': '80', 'filled_set': 'unfilled', 'dataset': 'train', 'dir': shifted_80_notfilled_train })
        shifted_80_notfilled_valid = os.path.join(shifted_80_dir_notfilled, 'valid')
        data.append({'degree': '80', 'filled_set': 'unfilled', 'dataset': 'valid', 'dir': shifted_80_notfilled_valid })
        
        dirlist = [ shifted_00_dir_filled, \
                    shifted_00_filled_train, \
                    shifted_00_filled_valid, \
                    shifted_00_dir_notfilled, \
                    shifted_00_notfilled_train, \
                    shifted_00_notfilled_valid, \
                    shifted_20_dir_filled, \
                    shifted_20_filled_train, \
                    shifted_20_filled_valid, \
                    shifted_20_dir_notfilled,\
                    shifted_20_notfilled_train, \
                    shifted_20_notfilled_valid, \
                    shifted_40_dir_filled, \
                    shifted_40_filled_train, \
                    shifted_40_filled_valid, \
                    shifted_40_dir_notfilled, \
                    shifted_40_notfilled_train, \
                    shifted_40_notfilled_valid, \
                    shifted_60_dir_filled, \
                    shifted_60_filled_train, \
                    shifted_60_filled_valid, \
                    shifted_60_dir_notfilled, \
                    shifted_60_notfilled_train, \
                    shifted_60_notfilled_valid, \
                    shifted_80_dir_filled, \
                    shifted_80_filled_train, \
                    shifted_80_filled_valid, \
                    shifted_80_dir_notfilled, \
                    shifted_80_notfilled_train, \
                    shifted_80_notfilled_valid ]

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        if not os.path.isdir(dir_path_filled):
            os.mkdir(dir_path_filled)
        if not os.path.isdir(dir_path_notfilled):
            os.mkdir(dir_path_notfilled)

        for _, dir in enumerate(dirlist):
            if not os.path.isdir(dir):
                os.mkdir(dir)

        dir_df = pd.DataFrame(data)
        dir_df_s = dir_df.astype(str)
        dir_df_s.to_csv(self.csv_prepared_dirs, index=False)
        return dir_df_s
     

    # Create combined RGB + depth images and save in dest_default

    def create_rgbd_images(self):
        if os.path.isfile(self.csv_master_dataset):
            self.logger.info('reading from ' + self.csv_master_dataset)
            return pd.read_csv(self.csv_master_dataset)

        saving_dir_filled = self.master_dir_filled # first generate master data dir (filled)
        if not os.path.isdir(saving_dir_filled):
            os.mkdir(saving_dir_filled)
        saving_dir_notfilled = self.master_dir_notfilled # and not filled
        if not os.path.isdir(saving_dir_notfilled):
            os.mkdir(saving_dir_notfilled)


        src_df = self.aggregate_frame_data()

        # create dirs for all categories
        categories = np.unique(src_df.category)

        for _, category in enumerate(categories):
            cat_dir = os.path.join(saving_dir_notfilled, category)
            if not os.path.isdir(cat_dir):
                os.mkdir(cat_dir)
            cat_dir = os.path.join(saving_dir_filled, category)
            if not os.path.isdir(cat_dir):
                os.mkdir(cat_dir)

        data = []

        for i in range(len(src_df.index)):
            current_rgb_row = src_df.iloc[[i]]

            current_category = current_rgb_row.category.values[0]
            current_instance_number = current_rgb_row.instance_number.values[0]
            current_video_no = current_rgb_row.video_no.values[0]
            current_frame_no = current_rgb_row.frame_no.values[0]
            dest_filename = current_rgb_row.dest_filename.values[0]
            current_dest_filename_filled = os.path.join(saving_dir_filled, current_category, current_rgb_row.dest_filename.values[0])
            current_dest_filename_notfilled = os.path.join(saving_dir_notfilled, current_category, current_rgb_row.dest_filename.values[0])
            current_crop_location = current_rgb_row.crop_location.values[0]
            current_depthcrop_location = current_rgb_row.depthcrop_location.values[0]


            self.logger.info("processing " + os.path.split(current_crop_location)[1]
                             + " and " + os.path.split(current_depthcrop_location)[1])

            # process image with filling
            self.preprocess_frame(current_crop_location, current_depthcrop_location, current_dest_filename_filled, fill=True)
            # process image without filling
            self.preprocess_frame(current_crop_location, current_depthcrop_location, current_dest_filename_notfilled, fill=False)

            data.append({
                'category': current_category,
                'instance_number': current_instance_number,
                'video_no': current_video_no,
                'frame_no': current_frame_no,
                'crop_location': current_crop_location,
                'depthcrop_location': current_depthcrop_location,
                'dest_filename': dest_filename,
                'master_filepath_filled': current_dest_filename_filled,
                'master_filepath_notfilled': current_dest_filename_notfilled
             })

        cur_df = pd.DataFrame(data)
        cur_df.to_csv(self.csv_master_dataset, index=False)
        return cur_df

    def get_process_dirs(self):
        if os.path.isfile(self.csv_process_dirs):
            self.logger.info('reading from ' + self.csv_process_dirs)
            return pd.read_csv(self.csv_process_dirs)

        images_df, struct_df = self.prepare_dest_dataset()
        self.logger.info('Get process dirs. Creating ' + self.csv_process_dirs)

        data = []

        for i in range(len(struct_df.index)):
            current_row = struct_df.iloc[[i]]

            current_degree = current_row.degree.values[0]
            current_filled_set = current_row.filled_set.values[0]
            current_dataset = current_row.dataset.values[0]
            if (current_dataset == 'valid'):
                continue
            train_dir = current_row.dir.values[0]

            valid_dir = struct_df[(struct_df.degree == current_degree) # direct access
                            & (struct_df.filled_set == current_filled_set)
                            & (struct_df.dataset == 'valid')].dir.values[0]

            data.append({'degree': current_degree, \
                        'filled_set': current_filled_set, \
                        'train_dir': train_dir, \
                        'valid_dir': valid_dir})

        dir_df = pd.DataFrame(data)
        dir_df.to_csv(self.csv_process_dirs, index=False)
        return dir_df

    def get_train_image_class_count(self):
        image_df, dir_df = self.prepare_dest_dataset()
        image_count = len(image_df[(image_df.set=='train')])
        
        class_count = len(image_df.category.unique())
        return image_count, class_count

    def get_valid_image_class_count(self):
        image_df, dir_df = self.prepare_dest_dataset()
        image_count = len(image_df[(image_df.set=='valid')])
        
        class_count = len(image_df.category.unique())
        return image_count, class_count


if __name__ == '__main__':
    ROOT_DEFAULT = '../rgbd-dataset'
    CSV_DATASET = '../rgbd-dataset_csv/rgbd-dataset.csv'
    CSV_AGGREGATED_DATASET = '../rgbd-dataset_csv/rgbd-dataset-aggregated.csv'
    CSV_MASTER_DATASET = '../rgbd-dataset_csv/rgbd-dataset-master.csv'
    CSV_TT_SPLIT_01 = '../rgbd-dataset_csv/rgbd-dataset-ttsplit01.csv'
    CSV_PREPARED_SET = '../rgbd-dataset_csv/rgbd-dataset-prepared.csv'
    CSV_PREPARED_DIRS = '../rgbd-dataset_csv/rgbd-dataset-prepared_dirs.csv'
    CSV_PROCESS_DIRS = '../rgbd-dataset_csv/rgbd-dataset-process_dirs.csv'
    MASTER_DEFAULT = '../master_dir'
    DEST_DEFAULT = '../data_to_process'
    DEST_SIZE = 224
    SPLIT_RATE = 0.2
  
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", default=ROOT_DEFAULT)
    parser.add_argument("--csv_dataset", default=CSV_DATASET)
    parser.add_argument("--csv_aggregated_dataset", default=CSV_AGGREGATED_DATASET)
    parser.add_argument("--csv_master_dataset", default=CSV_MASTER_DATASET)
    parser.add_argument("--csv_tt_split_01", default=CSV_TT_SPLIT_01)
    parser.add_argument("--csv_prepared_set", default=CSV_PREPARED_SET)
    parser.add_argument("--csv_prepared_dirs", default=CSV_PREPARED_DIRS)
    parser.add_argument("--csv_process_dirs", default=CSV_PROCESS_DIRS)
    parser.add_argument("--dest_default", default=DEST_DEFAULT)
    parser.add_argument("--master_default", default=MASTER_DEFAULT)
    parser.add_argument("--dest_size", default=DEST_SIZE)
    parser.add_argument("--split_rate", default=SPLIT_RATE)
    
    args = parser.parse_args()

    #f args.processed_data_output != '' and not os.path.isdir(args.processed_data_output):
    #    os.makedirs(args.processed_data_output)

    washington_dataset = WashingtonRGBD(root_dir=args.rootdir,
                                        csv_dataset=args.csv_dataset,
                                        csv_aggregated_dataset=args.csv_aggregated_dataset,
                                        csv_master_dataset=args.csv_master_dataset,
                                        csv_tt_split_01=args.csv_tt_split_01,
                                        csv_prepared_set=args.csv_prepared_set,
                                        csv_prepared_dirs=args.csv_prepared_dirs,
                                        csv_process_dirs=args.csv_process_dirs,
                                        master_default=args.master_default,
                                        dest_default=args.dest_default,
                                        dest_size=int(args.dest_size),
                                        split_rate=float(args.split_rate))
    # Hauptcode
    washington_dataset.load_metadata()
    washington_dataset.aggregate_frame_data()
    # build 2 master dirs
    washington_dataset.create_rgbd_images()

    # für das Ausprobieren einer einzelnen Konvertierung
    #washington_dataset.preprocess_frame('apple_1_1_1_crop.png', 'result.png', 'concat_test.png')
    #washington_dataset.preprocess_frame('binder_3_2_35_crop.png', 'binder_3_2_35_depthcrop.png', 'concat_test.png')
    # Gibt es gar nicht!!! - washington_dataset.preprocess_frame('binder_3_1_41_crop.png', 'binder_3_1_41_depthcrop.png', 'concat_test.png')
    # Gibt es gar nicht!!! - washington_dataset.preprocess_frame('binder_3_1_44_crop.png', 'binder_3_1_44_depthcrop.png', 'concat_test.png')

    # Splitting
    washington_dataset.train_test_split_01()
    
    # now prepare destination dataset
    washington_dataset.prepare_dest_dataset()
    # get process dirs
    washington_dataset.get_process_dirs()

    image_count, class_count = washington_dataset.get_train_image_class_count()
    washington_dataset.logger.info('Image count for training: ' + str(image_count))
    washington_dataset.logger.info('Class count for training: ' + str(class_count))

    
    washington_dataset.logger.info('Finished preparing')
    

