def Adjust_Data(img,mask,feature_dict, normalize):
    """
    Adjust image generator output

    Parameters: 
        -img: image array returned from generator (dim B x H x W x C)
        -mask: integer-labeled array returned from generator (dim B x H x W x 1)
        -feature_dict: dictionary that associates feature number and name
        -normalize: Boolean indicating whether image should be normalized

    Returns:
        -Tuple of adjusted image and mask (dim's (B x H x W x C))
    """

    if normalize:
        img = Normalize_Image(img)

    ## Flatten mask so it can be plugged into Expand_Mask
    mask = mask[:,:,:,0]
    mask = Expand_Mask(mask, feature_dict)
    return (img,mask)




def Crop_Image(img, x, y, width, height = None):
    """Crop image
    
    Parameters: 
        -img: image array (dim H x W x C)
        -x,y: upper left corner of crop rectangle
        -width, height: width and height of cropping rectangle
    
    Returns:
        -Cropped image of dimension H x W x C
    """
    
    if height == None:
        height = width
    
    img = img[y:y+height, x:x+width,:]
    return img


def custom_confusion_matrix(prediction_vector, true_vector, feature_dict ):
    """
    Create confusion matrix that works when prediction / labels do not contain 
    all feature classes. 
    
    Parameters: 
        -true_vector: true integer-valued labels (dim: N x 1)
        -prediction_vector: predicted labels (dim: N x 1)
        -feature_dict: dictionary that associates feature number and name
        
    Returns: 
        -Return F x F (F = number of features) confusion matrix even when certain classes are missing.
        Matrix is structured like Scikit Learn: C_ij is number of labels known to be in class
        i that are predicted to be in class j. 
    """
    
    values = list(feature_dict.keys())
    values.sort()
    nvals = len(values)
    confusion_matrix = np.zeros((nvals, nvals))
    for i in range(len(values)):
        for j in range(len(values)):
            mask = (true_vector==values[i]) & (prediction_vector==values[j]) 
            confusion_matrix[i,j] = mask.sum()
    
    return confusion_matrix


def Evaluate_Prediction(prediction_mask, true_mask, feature_dict, 
                        test_name = 'Test'):
    """
    Return dictionary containing metrics that evaluate image prediction and true mask
    and confusion matrix
    
    Parameters: 
        -prediction_mask: mask with pixels labeled by feature-number (dim H x W x 1)
        -true_mask: true image labels; mask with pixels labeled by feature-number (dim H x W x 1)
        -feature_dict: dictionary that associates feature number and name
        -test_name: value of "Test" column in dictionary output

    Returns: 
        -List containing dictionary and confusion matrix. 
    """
      
    # Convert mask to flat array (dim H x W) 
    true_mask = true_mask[:,:,0]
    
    # Convert from Prob to 0,1,2...
    prediction_mask = prediction_mask.argmax(axis = 2) 

    # Compute confusion matrix 
    conf = custom_confusion_matrix(prediction_mask.flatten(), true_mask.flatten(), feature_dict)
    
    # Convert mask to proper shape for loss function - shape should have 4 dimensions with one-hot encoding
    true_mask = Expand_Mask(mask = true_mask, num_class = len(feature_dict)) ## to 0,1
    true_mask = np.expand_dims(true_mask, axis=0)
    true_mask = true_mask.astype(np.float)

    # Convert prediction into proper shape for loss function
    prediction_mask = Expand_Mask(mask = prediction_mask, num_class = len(feature_dict)) #to 0,1
    prediction_mask = np.expand_dims(prediction_mask, axis=0)    
    prediction_mask = prediction_mask.astype(np.float)
    
    score = {'Test':test_name, 
             'Dice':Dice_Coef_Multilabel(true_mask, prediction_mask).numpy(), 
             'Accuracy':np.mean(tf.metrics.categorical_accuracy(true_mask, prediction_mask)), 
             'CE':np.mean(tf.metrics.categorical_crossentropy(true_mask, prediction_mask))}
    
    return [score, conf]
    
    
def Expand_Mask(mask, feature_dict):
    """
    Change flat masks into one-hot encoded multi-layer masks
    
    Parameters: 
        -mask: Array with pixels labeled by feature-number (dim H x W)
        -num_class: total number of different feature classes

    Returns:
        -One-hot encoded array of dimension H x W x F, where F is the number of features
    """
    
    new_mask = np.zeros(mask.shape + (len(feature_dict),))
    for i in feature_dict.keys():
        ni = int(i)
        new_mask[mask == ni,ni] = 1 
    return new_mask  


def Get_Label_Features(mask_in, feature_dict, convert_length = 0.2204315, eps_factor = 0.025, area_thresh = 2):
    """
    Convert image mask prediction into feature-contours
    
    Parameters:
        -mask_in: mask with pixels labeled by feature-number (dim H x W)
        -feature_dict: dictionary that associates feature number and name
        -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
        -eps_factor: Determines the wiggliness of contours
        -area_thresh: Feature is only kept if area is greater than [area_thresh] square meters
        
    Returns:
        -List of contour-lists (ordered as feature_dict) and pandas dataframe describing each feature  
    """    

    nfeatures = len(feature_dict)
    Image_Features = pd.DataFrame({'Type':[],'Feature_Area':[], 'x':[], 'y':[]})
    Contours_List = []

    #Expand mask into one-hot mask
    mask_in = Expand_Mask(mask_in, feature_dict = feature_dict)
    
    #Initialize list for storing contours
    for i in range(nfeatures):
        Contours_List.append(list()) 

    #Loop through mask layers (i.e., feature types) and calculate contours  
    for ii in feature_dict.keys():
        nii = int(ii)
        #Convert mask into a format that works with opencv
        mask = mask_in[:,:,nii]        
        mask = 255*mask.round().astype('uint8')
        mask = np.stack((mask,mask, mask),-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY);
        #Apply thresholding to determine contours
        ret, thresh = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #Loop through contours and extract feature-information
        for cnt in contours:
            # arcLength args: Contours, flag of whether curve is closed or not
            epsilon = eps_factor*cv2.arcLength(cnt,True)
            # approxPolyDP args: Contours, epsilon for wiggliness, closed shape or not
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            # Extract area and convert to square meters 
            area = convert_length**2 * cv2.contourArea(approx)
            if area > area_thresh: 
                ## Compute centroid from moments
                M = cv2.moments(cnt)
                cx = int(M['m10']/(1e-5 + M['m00']))*convert_length
                cy = int(M['m01']/(1e-5 + M['m00']))*convert_length
                Image_Features = Image_Features.append({'Type':feature_dict[ii], 'Feature_Area':area, 
                                                        'x':cx, 'y':cy}, ignore_index = True)
                Contours_List[nii].append(cnt)
    return Contours_List, Image_Features.copy()


def Get_Convolution_2(label, radius, feature_dict, pad = True, convert_length = 0.2204315, strides = 1, verbose = False, 
                   path = '', filename = '', meta = None):
    """ 
    Perform convolutions with circular arrays of 1's on a 2D label.

    Parameters: 
        -label: 2D array with pixels labeled by feature-number (dim H x W)
        -radius: radius  in meters of local habitat around Focal_Features to collect feature statistics over
        -feature_dict: dictionary that associates feature name and pixel number
        -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
        -strides: integer that describes stride length; used to increase speed (and reduce resolution of convolution). 
            Convolution is resized to original label dimension with opencv
        -verbose: print progress if True
        -Set the three arguments below to save a TIF raster of the convolved label
            -path: path to save directory when filename != ''
            -filename: name of file to save 
            -meta: metadata from raster used to create label 

    Returns: 
        -Numpy array (dim H x W x F) describing convolved label data. 
    """
    ## Make circular convolution filter at specified radius
    r = round(radius / convert_length)
    x = np.arange(0, 2*r)
    y = np.arange(0, 2*r)
    mask = (x[np.newaxis,:]-r)**2 + (y[:,np.newaxis]-r)**2 < r**2  
    mask = mask[:,:,np.newaxis, np.newaxis]
    mask_tensor = tf.constant(mask, tf.float32)
    
    #Loop through label classes and perform convolutions
    num_class = len(feature_dict)
    expanded_label = Expand_Mask(label, feature_dict)
    lab_shape = expanded_label.shape
    all_lab = np.zeros((lab_shape[0], lab_shape[1], num_class))
    for val in range(num_class): 
        ohe_layer = expanded_label[:,:,val]
        ohe_tensor = tf.constant(ohe_layer[np.newaxis, :, :, np.newaxis], tf.float32)
        tensor_res = tf.nn.convolution(ohe_tensor, mask_tensor, padding='SAME', strides = [strides]) 
        img_res = tensor_res[0,:,:,0].numpy()
        # Note that the label shape is reversed here (to preserve predicted_class shape)
        img_res = cv2.resize(img_res, (label.shape[1], label.shape[0]))
        all_lab[:,:,val] = img_res
        if verbose:
            print('Finished: ' + str(val))

    ## Convert output to square meters 
    all_lab = all_lab*convert_length**2
    
    #Save raster of convolved label data
    if filename !='':
        try:
            if path == '':
                path = 'Predictions'
            os.makedirs(path)
        except OSError as error: 
            print('')
        # Write raster label to file
        meta.update(count = num_class, nodata = 255, compress = 'deflate', predictor = 2)
        tif_lab = np.moveaxis(all_lab,-1,0)
        with rasterio.open(path + '/' + filename + '.tif', 'w', **meta) as src:
            src.write(tif_lab)  
    return all_lab

        
def Get_Predictors_At_XY_From_DF(Features, Image_Features, transform, 
                        convert_length = 0.2204315, radius = 50, Building_Cols = ['modern_build','trad_build']):
    """ 
    Extract building-count statistics from dataframe at specific Longitude/Latitude points. 

    Parameters: 
        -Features: dataframe describing location of points with (Long/Lat) point location in meters
        -Image_Features: dataframe describing (x,y) location and Type of each building (output of Get_Label_Features)
        -transform: index object from raster image
        -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
        -radius: radius of local habitat in meters around Focal_Features to collect feature statistics over
        -Building_Cols: List of columns in Image_Features that describe modern first, then traditional buildings
        
    Returns:
        -Copy of Features dataframe with added building count columns 
    """
   
    Focal_Features = Features.copy()

    ## Define factor by which to convert pixel area to area in square meters
    convert_area = convert_length**2
    pixel_radius = radius/convert_length
    
    ## Create mask and index list that shows which image_features are buildings
    mask_buildings = Image_Features.Type.isin(Building_Cols) 
    ind_buildings = list(mask_buildings[mask_buildings].index) 
    nbuildings = len(ind_buildings)
    Image_Features = Image_Features[mask_buildings]
    Image_Features = Image_Features.reset_index()

    ## Create submasks that distinguish modern buildings from huts 
    mask_mods = Image_Features.Type[mask_buildings].isin([Building_Cols[0]])
    ind_mods = list(mask_mods[mask_mods].index)
    mask_huts = Image_Features.Type[mask_buildings].isin([Building_Cols[1]])
    ind_huts = list(mask_huts[mask_huts].index)
        
    Focal_Features['x'], Focal_Features['y'] = transform(Focal_Features.Longitude, Focal_Features.Latitude)    
        
    ## Calculate distances between Focal features and buildings
    distance_mat = dist(Focal_Features.loc[:,{'x','y'}], 
                        Image_Features.loc[:,{'x','y'}])
    Focal_Features.loc[:, 'Local_Buildings'] = None
    Focal_Features.loc[:, 'Local_Moderns'] = None
    Focal_Features.loc[:, 'Local_Traditionals'] = None
    #print('u')    
    nfeatures = Focal_Features.shape[0]

    # Loop through each feature
    for ii in range(nfeatures):
        
        close_buildings = (distance_mat[ii, :] < pixel_radius).sum()
        close_mods = (distance_mat[ii, ind_mods] < pixel_radius).sum() 
        close_huts = (distance_mat[ii, ind_huts] < pixel_radius).sum() 
        Focal_Features.loc[ii, 'Local_Buildings'] = close_buildings    
        Focal_Features.loc[ii, 'Local_Moderns'] = close_mods           
        Focal_Features.loc[ii, 'Local_Traditionals'] = close_huts 

    return Focal_Features
    

def Get_Predictors_At_XY_From_Mask(Features, feature_dict, mask_in, transform):
    """ 
    Extract values from a raster at specific Longitude / Latitude coordinates 
    
    Parameters: 
        -Features: pandas dataframe describing Longitude/Latitude locations at which to extract values 
        -feature_dict: dictionary that associates feature name and pixel number
        -mask_in: array with bands corresponding to features in feature_dict
        -transform: index raster object that relates Longitude/Latitude to pixel coordinates 

    Return:
        -Features dataframe with extracted values added
    """ 
    Focal_Features = Features.copy()
    # Get coordinates of each Focal_Feature in terms of pixel coordinates
    ix,iy = transform(Focal_Features.Longitude, Focal_Features.Latitude)
    
    feature_list = list()
    # Extract features from convolutional mask if provided
    if mask_in is not None:
        for val in feature_dict.keys():
            ival = int(val)
            feature_name = 'Local_' + feature_dict[val] + '_Area'
            feature_list += [feature_name]
            Focal_Features.loc[:, feature_name] = mask_in[iy,ix, ival]  
            
    return Focal_Features.copy()   

        
#################### ---------- LOSS FUNCTIONS BELOW -------------- #####################



def Weighted_Cross_Entropy(y_true, y_pred, eps = 1e-10):
    """
    Calculate weighted cross entropy. Requires "weights" to be defined.
    
    Parameters:
        -y_true: OHE array of true values (dim B x H x W x F, F~ number of features)
        -y_pred: OHE array of probabilities assigned by model (dim B x H x W x F)
        -eps: smoothing factor
        
    Returns:
        -loss value   
    """
    y_pred = tf.cast(y_pred, 'float64')
    y_true = tf.cast(y_true, 'float64')
    #Use broadcasting to mask relevant weights
    class_weights = weights * y_true
    #Compute softmax cross entropy loss
    unweighted_losses = y_true*tf.math.log(y_pred + eps)
    weighted_losses = unweighted_losses * class_weights
    #Sum the result to get your final loss
    loss = -tf.reduce_sum(weighted_losses)
    return loss



def Dice_Coef_Multilabel(y_true, y_pred, numLabels=len(feature_dict), smooth = 1e-7):
    """
    Calculate dice coefficient across all channels of prediction.
    
    Parameters:
        -y_true: OHE array of true values (dim B x H x W x F, F~ number of features). Values of 255 are ignored
        -y_pred: OHE array of probabilities assigned by model (dim B x H x W x F)
        -index: channel for which loss is being calculated
        -eps: smoothing factor
        
    Returns:
        -loss value   
    """
    
    dice=0
    
    for index in range(numLabels):
        
        dice -= Dice_Coef(y_true[:,:,:,index], y_pred[:,:,:,index], smooth = smooth)     
        
    return dice   


def Dice_Coef(y_true_index, y_pred_index, smooth = 1e-7):
    """
    Calculate dice coefficient on single channel. 
    
    Parameters:
        -y_true_index: OHE array of true values (dim B x H x W x F, F~ number of features). Values of 255 are ignored
        -y_pred_index: OHE array of probabilities assigned by model (dim B x H x W x F)
        -index: channel for which loss is being calculated
        -eps: smoothing factor
        
    Returns:
        -loss value   
    """
    
    y_true_f = K.flatten(y_true_index)
    y_pred_f = K.flatten(y_pred_index)

    mask_include = (y_true_f!=255)
    y_true_f = y_true_f[mask_include]
    y_pred_f = y_pred_f[mask_include]
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2. * intersection + smooth) / (union + smooth)



#################### ---------- LOSS FUNCTIONS ABOVE -------------- #####################




def Normalize_Image(img):
    """
    Normalize image
    
    Parameters: image array (dim H x W x C)

    Returns: Normalized image
    """
    img = img / 127.5
    img = img - 1.0 
    return (img)  




def Predict_Class_Probabilities(array, model, step = 100, normalize = False, 
                                filename = None, path = ''):
    """
    Apply model to image and return image array of integer-coded class predictions.
    
    Parameters:
        -array: image array (dim H x W x C)
        -model: CNN model object
        -step: prediction window is moved a distance step after each prediction; multiple pixel predictions are averaged
        -normalize: option to normalize image before applying CNN (required for VGG16) 
        -filename: save location if not None
        -path: directory of save location
    
    Returns:
        -predicted_image: numpy array of prediction probabilities (dim H x W x F)
        -predicted_class: numpy array of class predictions (dim H x W)
    """
    
    img_height, img_width = model.input_shape[1:3]
    num_class = model.output_shape[3]
    image_orig = array.astype('float')
    if normalize:
        image_orig = Normalize_Image(image_orig)
    #Create batch dimension for compatibility with CNN    
    image_orig = np.expand_dims(image_orig, axis=0)
    shp = np.array(array.shape)
    shp[2] = num_class
    predicted_image = 0.0*np.zeros(shp).astype('float')
    count_image = 0.0*predicted_image
    #Slide prediction window across image
    for i in range(array.shape[0]//step + 1):
        starti = step*i
        endi = starti + img_height
        if(endi > array.shape[0]):
            starti = array.shape[0] - img_height
            endi = array.shape[0]
        for j in range(array.shape[1]//step + 1):
            startj = step*j
            endj = startj + img_width
            if(endj > array.shape[1]):
                startj = array.shape[1] - img_width
                endj = array.shape[1]
            subimage = image_orig[:,starti:endi, startj:endj, :]         
            predicted_subimage = model.predict(subimage)
            predicted_image[starti:endi, startj:endj,:] = predicted_image[starti:endi, startj:endj,:] + \
            predicted_subimage[0,:,:,:]
            count_image[starti:endi, startj:endj,:] = count_image[starti:endi, startj:endj,:] + 1.0
    #Average pixel predictions for pixels with multiple predictions
    predicted_image = np.divide(predicted_image,count_image); 
    predicted_class = predicted_image.argmax(axis = 2)    
    
    #Save prediction
    if filename is not None:
        try:
            if path == '':
                path = 'Predictions'
            os.makedirs(path)
        except OSError as error: 
            print('') 
        fig, ax = plt.subplots(figsize=(18, 20))
        ax.imshow(predicted_class)
        plt.tight_layout()
        plt.savefig(path + '/' + filename + '_Predict_Mask.png', bbox_inches='tight')
        plt.close(fig)
    return predicted_image, predicted_class
    
        
def Predict_Image_Contours(img, mask_full, feature_dict, area_thresh = 30, filename = None, path = ''):
    """ 
    Takes image and mask as input and returns image with mask contours overlaid
        
    Parameters:
        -img: image array
        -mask_full: expanded, one-hot-encoded mask (dim H x W x F; F~number features)
        -feature_dict: dictionary that associates feature number and name
        -area_thresh: features smaller than this are not retained (pixel-based area)
        -filename: save location if not None
        -path: directory of save location
    
    Returns:
        -modified image (dim H x W x C)   
    """
    # Loop through features
    for ii in range(len(feature_dict)):
        Type = feature_dict[str(ii)]
        if Type=='modern_build':
            color_rgb  = (1.0,0,0)
        elif Type=='trad_build':
            color_rgb = (0,0,1.0)

        # Modify mask so it can be plugged into opencv FindContours 
        mask = mask_full[:,:,ii]
        mask = 255*mask.round().astype('uint8')
        mask = np.stack((mask,mask, mask),-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY);
        ret, thresh = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through features and overlay on image
        for cnt in contours:
            ## Contours, flag of whether curve is closed or not
            epsilon = 0.025*cv2.arcLength(cnt,True)
            ## Contours, epsilon for wiggliness, closed shape or not
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            ## Extract Area Dest image, contours, contour index
            area = cv2.contourArea(approx)
            ## centroid computed from moments
            M = cv2.moments(cnt)    
            if area > area_thresh:
                if Type=='modern_build':
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    img = cv2.drawContours(image = img, contours = [box], 
                                      contourIdx = 0, color = color_rgb, 
                                      thickness = 2)
                elif Type=='trad_build':
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x),int(y))
                    radius = int(radius)
                    img = cv2.circle(img,center,radius,color_rgb,2)
                elif Type=='Forest':
                    img = cv2.drawContours(image = img, contours = [cnt], 
                                      contourIdx = 0, color = color_rgb, 
                                      thickness = 2)
                elif Type=='Bare':
                    img = cv2.drawContours(image = img, contours = [cnt], 
                                      contourIdx = 0, color = color_rgb, 
                                      thickness = 2)
    if filename is not None:
        try: 
            if path == '':
                path = 'Predictions'
            os.makedirs(path)
        except OSError as error: 
            print('') 
        fig, ax = plt.subplots(figsize=(18, 20))
        ax.imshow(img[:,:,0:3])
        plt.tight_layout()
        plt.savefig(path + '/' + filename, bbox_inches='tight')                   
        plt.close(fig)
                   
    return img



def Save_Image_Crop(img, x, y, width, height = None, filename = None, path = 'Predictions'):
    """
    Crop image and save PNG
    
    Parameters: 
        -img: image
        -x,y: upper left corner of crop rectangle
        -width, height: width and height of crop rectangle
        -filename, path: save location of cropped image
    """
    
    if height is None:
        height = width
        
    img = img[y:y+height, x:x+width,:]

    if filename is not None:
        try: 
            os.mkdir(path)
        except OSError as error: 
            print('') 
        fig, ax = plt.subplots(figsize=(18, 20))
        ax.imshow(img)
        plt.tight_layout()
        plt.savefig(path + '/' + filename + '_Crop.png')


def Summarize_Features(Features):
    """
    Summarize output from Count_Features function into dataframe
    
    Parameters:
        -Features: output of Get_Label_Features. Pandas dataframe with Type column that describes buildings. 
        
    Return:
        -pandas dataframe with summarized building counts
    
    """
    Building_Features = Features[Features.Type.isin(['modern_build','trad_build'])]
    nbuildings = Building_Features.shape[0]
    if nbuildings > 0:
        nModBuild = sum(Building_Features.Type=='modern_build')
        nTradBuild = sum(Building_Features.Type=='trad_build')
        fracmBuild = np.round(nModBuild / nbuildings,3)
        fractBuild = 1 - fracmBuild
    else: 
        nModBuild = 0
        nTradBuild = 0      
        fracmBuild = 0
        fractBuild = 0
    summ = pd.DataFrame({'nModBuild':nModBuild,
                         'nTradBuild':nTradBuild,
                         'fracMBuild':[fracmBuild], 'fracTBuild':[fractBuild]})
    return summ
        


class DataGenerator(tf.keras.utils.Sequence):
    
    """
    Create an image data generator for segmentation maps
    
    Parameters: 
        -df: dataframe with image and label columns describing respective file locations
        -seq: imgaug Sequence object that describes types of augmentation
        -x_col: column name that contains image file locations
        -y_col: column name that contains label file locations
        -feature_dict: dictionary that associates feature name and pixel number
        -batch_size
        -normalize: True/False whether to scale images
        -shuffle: True/False whether to shuffle images after each epoch
        
    Returns:
        -batch of images and segmentation maps 
    """
    def __init__(self, df, seq, x_col, y_col, feature_dict, batch_size=32, normalize = False, shuffle=True):
        self.df = df
        self.seq = seq
        self.indices = self.df.index.tolist()
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.feature_dict = feature_dict
        self.batch_size = batch_size
        self.normalize = normalize
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
        
    def __len__(self):
        return len(self.indices) // self.batch_size

    # Get train and label data
    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    # Function to apply after each iteration
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    # Load images and apply augmentation
    def __get_data(self, batch):
        ## Load images
        image_list = [imageio.imread(img, pilmode = 'RGB')[np.newaxis, :, :, :] 
              for img in self.df[self.x_col].iloc[batch]]
        image_batch = np.concatenate(image_list, axis = 0)

        ## Load segmentation masks
        label_list = list()
        for img in self.df[self.y_col].iloc[batch]:
            image_array = imageio.imread(img, pilmode = 'RGB')[:,:,0]
            exp_image_array = Expand_Mask(image_array, self.feature_dict)
            exp_image_array = exp_image_array[np.newaxis, :, :, :]
            label_list.append(exp_image_array)
        
        # Label loaded as 3 channel RGB but only 1 channel of data - discard redundant channels
        label_batch = np.concatenate(label_list, axis = 0).astype('uint8')
        
        ## Apply augmentation
        image_batch, label_batch = self.seq(images = image_batch, 
                            segmentation_maps = label_batch)

        
        ## Normalize
        if self.normalize:
            image_batch = (image_batch/127.5 - 1.0)
                
        return image_batch, label_batch
    
    # Yield next batch
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result






    
############################
############################   
############################   

  
    
########## ----------------------- DEPRECATED -----------------------------##############


## Deprecated
def Get_Mask_Predictors(mask_in, Image_Features, feature_dict, 
                        convert_length = 0.2204315, radius = 50, verbose = False):
    """ 
    Extract predictors from image_features dataframe (output of Get_Mask_Features). Finds the local area of forest, bare, huts, and modern buildings within radial distance of each focal building. Returns Image_Features dataframe with added feature columns
    
    Parameters: 
        -mask_in: mask with pixels labeled by feature-number (dim H x W)
        -Image_Features: dataframe describing (x,y) location and Type of each building (output of Get_Label_Features)
        -feature_dict: dictionary that associates feature name and pixel number
        -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
        -radius: radius of local habitat in meters around Focal_Features to collect feature statistics over
        -verbose: print progress if True
    
    Return:
        -
    """    
    
    ## Expand mask into one-hot mask if input is flat
    if len(mask_in.shape)==2:
        mask_in = Expand_Mask(mask_in, num_class = nfeatures)
    
    ## Define factor by which to convert pixel area to area in square meters
    convert_area = convert_length**2
    
    ## create mask and index list that shows which image_features are buildings
    mask_buildings = Image_Features.Type.isin(['mBuild', 'tBuild']) 
    ind_buildings = list(mask_buildings[mask_buildings].index) 
    nbuildings = len(ind_buildings)
    
    ## Create submasks that distinguish modern buildings from huts 
    mask_mods = Image_Features.Type[mask_buildings].isin(['mBuild'])
    ind_mods = list(mask_mods[mask_mods].index)
    mask_huts = Image_Features.Type[mask_buildings].isin(['tBuild'])
    ind_huts = list(mask_huts[mask_huts].index)
    
    ## Calculate distances between all buildings
    distance_mat = dist(Image_Features.loc[ind_buildings,{'x','y'}])
    
    Image_Features.loc[:, 'Local_Buildings'] = None
    Image_Features.loc[:, 'Local_Moderns'] = None
    Image_Features.loc[:, 'Local_Traditionals'] = None
    Image_Features.loc[:, 'Local_Forest_Area'] = None
    Image_Features.loc[:, 'Local_Bare_Area'] = None
    Image_Features.loc[:, 'Local_Modern_Area'] = None
    Image_Features.loc[:, 'Local_Trads_Area'] = None
    Image_Features.loc[:, 'Local_Focal_Area'] = 3.14159*radius**2
    
    # Loop through each building and collect statistics
    for ii in ind_buildings:
        ind = ind_buildings[ii]
        building_type = Image_Features.Type[ind]
        close_buildings = (distance_mat[ii, :] < radius).sum() - 1.0
        close_mods = (distance_mat[ii, ind_mods] < radius).sum() - 1.0*(building_type=='mBuild')
        close_huts = (distance_mat[ii, ind_huts] < radius).sum() - 1.0*(building_type=='tBuild')
        Image_Features.loc[ind, 'Local_Buildings'] = close_buildings    
        Image_Features.loc[ind, 'Local_Moderns'] = close_mods           
        Image_Features.loc[ind, 'Local_Traditionals'] = close_huts 
        ## Define mask that will select a circle around the focal building. Note
        ## that 0 and 1 indices of mask / image correspond to rows (y) and cols (x)
        x = np.arange(0, mask_in.shape[1])
        y = np.arange(0, mask_in.shape[0])
        ## Convert distances back into pixels
        cx = round(Image_Features.loc[ind, 'x'] / convert_length)
        cy = round(Image_Features.loc[ind, 'y'] / convert_length)
        r = (radius / convert_length)
        ## Make indicator mask of all pixels less than distance r from focal building
        mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2   
        Image_Features.loc[ind, 'Local_Modern_Area'] = mask_in[mask,0].sum()*convert_area 
        Image_Features.loc[ind, 'Local_Trads_Area'] = mask_in[mask,1].sum()*convert_area  
        Image_Features.loc[ind, 'Local_Forest_Area'] = mask_in[mask,2].sum()*convert_area 
        Image_Features.loc[ind, 'Local_Bare_Area'] = mask_in[mask,3].sum()*convert_area   
        if verbose:
            print(str(ii) + ' / ' + str(nbuildings))
    return Image_Features




#Deprecated - replaced with Get_Convolution_2
def Get_Convolution(label, radius, feature_dict, pad = True, convert_length = 0.2204315, verbose = False, 
                   path = '', filename = '', meta = None):
    """ Perform convolutions with circular arrays of 1's on a 2D label.  

    Parameters: 
        -label: 2D array with pixels labeled by feature-number
        -radius: radius of local habitat in meters around Focal_Features to collect feature statistics over
        -feature_dict: dictionary that associates feature name and pixel number
        -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
        -verbose: print progress if True

    """
    ## Make convolution at specified radius
    r = round(radius / convert_length)
    num_class = len(feature_dict)
    ## Create circular filter window
    x = np.arange(0, 2*r)
    y = np.arange(0, 2*r)
    mask = (x[np.newaxis,:]-r)**2 + (y[:,np.newaxis]-r)**2 < r**2  
    mask = mask[:,:,np.newaxis, np.newaxis]
    mask_tensor = tf.constant(mask, tf.float32)

    expanded_label = Expand_Mask(label, feature_dict)
    lab_shape = expanded_label.shape
    all_lab = np.zeros((lab_shape[0] - mask.shape[0] + 1, lab_shape[1] - mask.shape[1] + 1, num_class))
    for val in range(num_class): 
        ohe_layer = expanded_label[:,:,val]
        ohe_tensor = tf.constant(ohe_layer[np.newaxis, :, :, np.newaxis], tf.float32)
        tensor_res = tf.nn.convolution(ohe_tensor, mask_tensor, padding='VALID') 
        all_lab[:,:,val] = tensor_res.numpy()[0,:,:,0]
        if verbose:
            print('Finished: ' + str(val))
            
    if pad:
        array_shape = label.shape
        # up-down padding
        tot_pw_ud = (array_shape[0] - all_lab.shape[0])/2
        pw_up = int(np.ceil(tot_pw_ud))
        pw_down = int(np.floor(tot_pw_ud))
        # left-right padding
        tot_pw_lr = (array_shape[1] - all_lab.shape[1])/2
        pw_left = int(np.ceil(tot_pw_lr))
        pw_right = int(np.floor(tot_pw_lr))
        all_lab_pad = np.pad(all_lab, pad_width = ((pw_down, pw_up), (pw_left, pw_right), (0,0)), 
           mode = 'constant', constant_values = 255)
    
    if filename !='':
        try:
            if path == '':
                path = 'Predictions'
            os.makedirs(path)
        except OSError as error: 
            print('')     
        
        meta.update(count = num_class, nodata = 255, compress = 'deflate', predictor = 2)
        
        # Write raster label to file
        tif_lab_pad = np.moveaxis(all_lab_pad,-1,0)
        with rasterio.open(path + '/' + filename + '.tif', 'w', **meta) as src:
            src.write(tif_lab_pad)  
    return all_lab_pad

## Deprecated; use Data_Generator
def Image_Generator(gen_dataframe, feature_dict, rot = 360, batch_size = 5, image_color_mode = "rgb",
                    mask_color_mode = "grayscale", zoom_range = 0.2, 
                    horizontal_flip = True, vertical_flip = True, fill_mode = 'reflect',
                    target_size = (512,512), seed = 1, normalize = False, image_col = 'Image', 
                   label_col = 'Label', adjust = True):
    '''
    Wrapper for image_datagen methods for data augmentation 
    
    Parameters:
        -gen_dataframe: dataframe with image and label columns describing respective file locations
        -feature_dict: dictionary that associates feature name and pixel number
        -rot: rotation
        -batch_size
        -image_color_mode
        -mask_color_mode
        -zoom_range
        -horizontal, vertical flip
        -fill_mode
        -target_size
        -seed 
        -normalize: boolean value indicating whether to normalize the images after augmentation
        -image_col: column of gen_dataframe that describes image file location
        -label_col: column of gen_dataframe that describes segmentation map file location
        -adjust: whether to apply post-augmentation "Adjust_Image" script
        -imgaug: whether to implement augmentation "seq" sequence from ImgAug
        
    Returns: 
        -iteratable generator that returns image and mask, each in the format (B x H x W x C)
    '''
    
    image_datagen = ImageDataGenerator(rotation_range=rot, width_shift_range=0.0,
                                       height_shift_range=0.0,
                                       zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip, vertical_flip = vertical_flip,
                                       fill_mode=fill_mode)

    image_generator = image_datagen.flow_from_dataframe(
        gen_dataframe, x_col = image_col,
        classes = None,
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    mask_generator = image_datagen.flow_from_dataframe(
        gen_dataframe, x_col = label_col,
        classes = None,
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)

    ## Combine the image and mask into a single output
    combined_generator = zip(image_generator, mask_generator)    
    for (img,mask) in combined_generator:
        if adjust:
            img, mask = Adjust_Data(img,mask,feature_dict, normalize)
        yield (img,mask)        
