import numpy as np

class DataGeneratorUNet2D():
    'Generates batch data for Keras'
    def __init__(self, input_shape = (256, 256, 2), output_shape = (256, 256, 1), case_dir = '', batch_size = 40, shuffle = False):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.case_dir = case_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, list_IDs):
        while 1:
            case_dir = self.case_dir
            batch_size = self.batch_size
            imax = int(np.ceil(len(list_IDs)/batch_size))

            for l in range(imax):
                # Find list of IDs
                temp_ID = []
                for i in list_IDs[l*batch_size : (l+1)*batch_size]:
                    temp_ID.append(i)
                
                indexes = self.__get_exploration_order(temp_ID)
                X, y = self.__data_generation( temp_ID, indexes, batch_size, case_dir)
                yield X, y
            
    def __get_exploration_order(self, list_IDs):
        'Updates indexes (exploration order) after each epoch'
        indexes = np.arange(len(list_IDs))
        #indexes = list_IDs
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, listIDs, indexes, batch_size, case_dir):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.empty((self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        
        # Generate data
        for i in indexes:
            temp_CTA = np.load(case_dir + 'images/' + listIDs[i][:-4] + '_CTA.npy')
            
            # compute center offset
            x_center = (self.input_shape[0] - temp_CTA.shape[0]) // 2
            y_center = (self.input_shape[1] - temp_CTA.shape[1]) // 2
            
            X[i,x_center:x_center+temp_CTA.shape[0],y_center:y_center+temp_CTA.shape[1],0] = temp_CTA
            X[i,x_center:x_center+temp_CTA.shape[0],y_center:y_center+temp_CTA.shape[1],1] = np.load(case_dir + 'images/' + listIDs[i][:-4] + '_NCCT.npy')
            y[i,x_center:x_center+temp_CTA.shape[0],y_center:y_center+temp_CTA.shape[1],0] = np.load(case_dir + 'masks/' + listIDs[i]) 

        return X, y
