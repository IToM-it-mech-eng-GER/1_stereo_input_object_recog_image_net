from keras.preprocessing.image import ImageDataGenerator

RGBD = 'rgbd'
NORB = 'norb'
SEED = 42

class Generator(object):
    def __init__(self, seed=0):
        self.seed = seed
        self.generator_l_t = None
        self.generator_r_t = None
        self.generator_l_v = None
        self.generator_r_v = None
        self.image_datagen_left_t = None
        self.image_datagen_left_v = None
        self.image_datagen_right_t = None
        self.image_datagen_right_v = None
        self.train_generator = None
        self.valid_generator = None
    
    def get_classes(self):
        return self.generator_l_v.classes
 
    def get_class_indices(self): # 
        return self.generator_l_v.class_indices
    
    def get_filenames(self):
        return self.generator_l_v.filenames
    
    def get_index_array(self):
        return self.generator_l_v.index_array
 
    # define data generators
    def double_data_generator_t(self, dataset, image_dir_l, image_dir_r, batch_size):
        if (dataset == RGBD):    
            self.generator_l_t = self.image_datagen_left_t.flow_from_directory(image_dir_l,target_size=(224, 448),batch_size=batch_size,class_mode='categorical',seed=SEED) 
            self.generator_r_t = self.image_datagen_right_t.flow_from_directory(image_dir_r,target_size=(224, 448),batch_size=batch_size,class_mode='categorical',seed=SEED)
        else: # NORB
            self.generator_l_t = self.image_datagen_left_t.flow_from_directory(image_dir_l,target_size=(224, 224),batch_size=batch_size,class_mode='categorical',seed=SEED) 
            self.generator_r_t = self.image_datagen_right_t.flow_from_directory(image_dir_r,target_size=(224, 224),batch_size=batch_size,class_mode='categorical',seed=SEED)

        while True:
            X_left = self.generator_l_t.next()
            X_right = self.generator_r_t.next()
            yield [X_left[0], X_right[0]], X_left[1]  #Yield both images and their mutual label
    
    def double_data_generator_v(self, dataset, image_dir_l, image_dir_r, batch_size):
        if (dataset == RGBD):    
            self.generator_l_v = self.image_datagen_left_v.flow_from_directory(image_dir_l,target_size=(224, 448),batch_size=batch_size,class_mode='categorical',seed=SEED) 
            self.generator_r_v = self.image_datagen_right_v.flow_from_directory(image_dir_r,target_size=(224, 448),batch_size=batch_size,class_mode='categorical',seed=SEED)
        else: # NORB
            self.generator_l_v = self.image_datagen_left_v.flow_from_directory(image_dir_l,target_size=(224, 224),batch_size=batch_size,class_mode='categorical',seed=SEED) 
            self.generator_r_v = self.image_datagen_right_v.flow_from_directory(image_dir_r,target_size=(224, 224),batch_size=batch_size,class_mode='categorical',seed=SEED)

        while True:
            X_left = self.generator_l_v.next()
            X_right = self.generator_r_v.next()
            yield [X_left[0], X_right[0]], X_left[1]  #Yield both images and their mutual label
        
    def single_data_generator_t(self, dataset, image_dir_l, batch_size):
        
        if (dataset == RGBD):
            self.generator_l_t = self.image_datagen_left_t.flow_from_directory(image_dir_l,target_size=(224, 448),batch_size=batch_size,class_mode='categorical',seed=SEED) 
        else:
            self.generator_l_t = self.image_datagen_left_t.flow_from_directory(image_dir_l,target_size=(224, 224),batch_size=batch_size,class_mode='categorical',seed=SEED) 

        while True:
            X_left = self.generator_l_t.next()
            yield X_left[0], X_left[1]  #Yield image and its label

    def single_data_generator_v(self, dataset, image_dir_l, batch_size):
        
        if (dataset == RGBD):
            self.generator_l_v = self.image_datagen_left_v.flow_from_directory(image_dir_l,target_size=(224, 448),batch_size=batch_size,class_mode='categorical',seed=SEED) 
        else:
            self.generator_l_v = self.image_datagen_left_v.flow_from_directory(image_dir_l,target_size=(224, 224),batch_size=batch_size,class_mode='categorical',seed=SEED) 

        while True:
            X_left = self.generator_l_v.next()
            #print('.')
            yield X_left[0], X_left[1]  #Yield image and its label

    # new load data generators
    def load_data_generators(self, dataset, doublehead, train_dir_l, valid_dir_l, train_dir_r, valid_dir_r, batch_size):
            
            self.image_datagen_left_t = ImageDataGenerator(rescale=1./255)
            self.image_datagen_left_v = ImageDataGenerator(rescale=1./255)
            self.image_datagen_right_t = ImageDataGenerator(rescale=1./255)
            self.image_datagen_right_v = ImageDataGenerator(rescale=1./255)
            
            if (doublehead):
                # coupled generators for double headed net 
                self.train_generator = self.double_data_generator_t(dataset, train_dir_l, train_dir_r, batch_size)
                self.valid_generator = self.double_data_generator_v(dataset, valid_dir_l, valid_dir_r, batch_size)
            else:
                # normal generators for single net
                self.train_generator = self.single_data_generator_t(dataset, train_dir_l, batch_size)
                self.valid_generator = self.single_data_generator_v(dataset, valid_dir_l, batch_size)
        
            #return train_generator, valid_generator
