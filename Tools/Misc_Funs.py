def Expand_Mask(mask, feature_dict):
    """Change flat masks into one-hot encoded multi-layer masks
    
    Parameters: 
    mask: mask with pixels labeled by feature-number
    num_class: total number of different feature classes
    
    """
    new_mask = np.zeros(mask.shape + (len(feature_dict),))
    for i in feature_dict.keys():
        ni = int(i)
        new_mask[mask == ni,ni] = 1 
    return new_mask  
    
def Adjust_Data(img,mask,feature_dict, normalize):
    """Adjust image generator output"""
    ## Normalize image
    if normalize:
        img = Normalize_Image(img)

    ## Assume mask shape has 4 dimensions - mask is (batch, x, y, color-channel)
    ## color-channels are redundant, so just choose the first. 
    mask = mask[:,:,:,0]
    
    ## Image_datagen performs interpolation when rotating, resulting in non-integer
    ## mask values. Round these back to integers before expanding the mask. 
    mask = mask.round() 
    mask = Expand_Mask(mask, feature_dict)
    #print(mask.shape, np.unique(mask, axis = 0))
    return (img,mask)

## 


def Crop_Image(img, mask, x, y, width, height):
    """Crop image and mask
    
    Parameters: 
    -img: image
    -mask: mask with pixels labeled by feature-number
    -x,y: bottom left corner of crop rectangle
    -width, height: width and height of crop rectangle
    """
    img = img[y:y+height, x:x+width,:]
    mask = mask[y:y+height, x:x+width,:]
    return img, mask


def custom_confusion_matrix(prediction_vector, true_vector, feature_dict ):
    """
    Return num_class x num_class confusion matrix even when certain classes are missing.
    Matrix is structured like Scikit Learn: C_ij is number of labels known to be in class
    i that are predicted to be in class j. 
    
    Parameters: 
        -true_vector: true labels
        -prediction_vector: predicted labels
        -feature_dict: dictionary that associates feature number and name
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
    
    Parameters: 
    -prediction_mask: mask with pixels labeled by feature-number
    -true_mask: true image labels; mask with pixels labeled by feature-number
    -feature_dict: dictionary that associates feature number and name
    -test_name: value of "Test" column in dictionary output
    """
      
    # true_mask has 3 layers but they are redundant
    true_mask = true_mask[:,:,0]
    
    # Convert from Prob to 0,1,2...
    prediction_mask = prediction_mask.argmax(axis = 2) + 1 

    # Compute confusion matrix -- subtract 1 so that first label is "0"   
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
    
    

def Get_Label_Features(mask_in, feature_dict, convert_length = 0.2204315, eps_factor = 0.025, area_thresh = 2):
    """
    Convert image mask prediction as features described by contours
    
    Returns contours_list (ordered as feature_dict) and image_dataframe describing each feature    
    
    Parameters:
    -mask_in: mask with pixels labeled by feature-number (2D)
    -feature_dict: dictionary that associates feature number and name
    -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
    -eps_factor: Determines the wiggliness of contours
    -area_thresh: Feature is only kept if area is greater than [area_thresh] square meters
    """    

    nfeatures = len(feature_dict)
    Image_Features = pd.DataFrame({'Type':[],'Feature_Area':[], 'x':[], 'y':[]})
    Contours_List = []

    # Expand mask into one-hot mask if input is flat
    if len(mask_in.shape)==2:
        mask_in = Expand_Mask(mask_in, feature_dict = feature_dict)
    
    # Loop through mask layers (i.e., feature types) and calculate contours    
    for i in range(nfeatures):
        Contours_List.append(list()) 
    for ii in feature_dict.keys():
        nii = int(ii)
        mask = mask_in[:,:,nii]        
        mask = 255*mask.round().astype('uint8')
        mask = np.stack((mask,mask, mask),-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY);
        ret, thresh = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # arcLength args: Contours, flag of whether curve is closed or not
            epsilon = eps_factor*cv2.arcLength(cnt,True)
            # approxPolyDP args: Contours, epsilon for wiggliness, closed shape or not
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            # Extract area and convert to square meters 
            area = convert_length**2 * cv2.contourArea(approx)
            if area > area_thresh: ## Filter small features / noise
                ## Compute centroid from moments
                M = cv2.moments(cnt)
                cx = int(M['m10']/(1e-5 + M['m00']))*convert_length
                cy = int(M['m01']/(1e-5 + M['m00']))*convert_length
                Image_Features = Image_Features.append({'Type':feature_dict[ii], 'Feature_Area':area, 
                                                        'x':cx, 'y':cy}, ignore_index = True)
                Contours_List[nii].append(cnt)
    return Contours_List, Image_Features.copy()


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
                

## Get_Predictors_At_XY -- Loops through TRAPS and extracts nearby features
def Get_Predictors_At_XY(Focal_Features, Image_Features, feature_dict, mask_in = None,
                        convert_length = 0.2204315, radius = 50, verbose = False):
    """ Extract predictors from trap dataframe (output of V1 Prep_Data.r). Find the local area of forest, bare, huts, and modern buildings within radial distance of each focal building. Returns Focal_Features dataframe with added feature columns

    Parameters: 
    -mask_in: mask with pixels labeled by feature-number
    -Focal_Features: dataframe describing location of traps with x, y columns denoting pixel location
    -Image_Features: dataframe describing location of each building (output of Get_Label_Features)
    -feature_dict: dictionary that associates feature name and pixel number
    -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
    -radius: radius of local habitat in meters around Focal_Features to collect feature statistics over
    -verbose: print progress if True
    """
   
    ## Expand mask into one-hot mask if input is flat
    #print('z')
    ## Define factor by which to convert pixel area to area in square meters
    convert_area = convert_length**2
    #print('y')
    ## create mask and index list that shows which image_features are buildings
    mask_buildings = Image_Features.Type.isin(['mBuild', 'tBuild']) 
    ind_buildings = list(mask_buildings[mask_buildings].index) 
    nbuildings = len(ind_buildings)
    #print('x')
    ## Create submasks that distinguish modern buildings from huts 
    mask_mods = Image_Features.Type[mask_buildings].isin(['mBuild'])
    ind_mods = list(mask_mods[mask_mods].index)
    mask_huts = Image_Features.Type[mask_buildings].isin(['tBuild'])
    ind_huts = list(mask_huts[mask_huts].index)
    #print('w')
        
    ## Calculate distances between Focal features and buildings
    distance_mat = dist(Focal_Features.loc[:,{'x','y'}], 
                        Image_Features.loc[ind_buildings,{'x','y'}])
    #print('v')    
    Focal_Features.loc[:, 'Local_Buildings'] = None
    Focal_Features.loc[:, 'Local_Moderns'] = None
    Focal_Features.loc[:, 'Local_Traditionals'] = None
    Focal_Features.loc[:, 'Local_Focal_Area'] = 3.14159*radius**2
    #print('u')    
    nfeatures = Focal_Features.shape[0]

    ## Extract features from convolutional mask if provided
    if mask_in is not None:
        for val in feature_dict.keys():
            ival = int(val)
            feature_name = 'Local_' + feature_dict[val] + '_Area'
            Focal_Features.loc[:, feature_name] = mask_in[iy, ix, ival]*convert_area  
    
    # Loop through each feature
    for ii in range(nfeatures):
        
        close_buildings = (distance_mat[ii, :] < radius).sum()
        close_mods = (distance_mat[ii, ind_mods] < radius).sum() 
        close_huts = (distance_mat[ii, ind_huts] < radius).sum() 
        #print('b' + str(ii))
        Focal_Features.loc[ii, 'Local_Buildings'] = close_buildings    
        Focal_Features.loc[ii, 'Local_Moderns'] = close_mods           
        Focal_Features.loc[ii, 'Local_Traditionals'] = close_huts 
        if verbose:
            print(str(ii) + ' / ' + str(nfeatures))
    return Focal_Features
    


def Get_Mask_Predictors(mask_in, Image_Features, feature_dict, 
                        convert_length = 0.2204315, radius = 50, verbose = False):
    """ Extract predictors from image_features dataframe (output of Get_Mask_Features). Finds the local area of forest, bare, huts, and modern buildings within radial distance of each focal building. Returns Image_Features dataframe with added feature columns
    
    Parameters: 
    -mask_in: mask with pixels labeled by feature-number
    -Image_Features: dataframe describing location of each building
    -feature_dict: dictionary that associates feature name and pixel number
    -convert_length: converts pixel length to meters when resolution is 2e-6 decimal degrees in QGIS
    -radius: radius of local habitat in meters around Focal_Features to collect feature statistics over
    -verbose: print progress if True
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
        ##print('b' + str(ii))
        Image_Features.loc[ind, 'Local_Buildings'] = close_buildings    
        Image_Features.loc[ind, 'Local_Moderns'] = close_mods           
        Image_Features.loc[ind, 'Local_Traditionals'] = close_huts 
        ##print('c' + str(ii))       
        ## Define mask that will select a circle around the focal building. Note
        ## that 0 and 1 indices of mask / image correspond to rows (y) and cols (x)
        x = np.arange(0, mask_in.shape[1])
        y = np.arange(0, mask_in.shape[0])
        ##print('d' + str(ii))
        ## Convert distances back into pixels
        cx = round(Image_Features.loc[ind, 'x'] / convert_length)
        cy = round(Image_Features.loc[ind, 'y'] / convert_length)
        r = (radius / convert_length)
        ##print('e' + str(ii))
        ## Make indicator mask of all pixels less than distance r from focal building
        mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2   
        ##print('f' + str(ii))        
        ##print('mask' + str(mask.shape) + 'mask_in' + str(mask_in.shape))
        Image_Features.loc[ind, 'Local_Modern_Area'] = mask_in[mask,0].sum()*convert_area 
        ##print('g' + str(ii))        
        Image_Features.loc[ind, 'Local_Trads_Area'] = mask_in[mask,1].sum()*convert_area  
        ##print('h' + str(ii))        
        Image_Features.loc[ind, 'Local_Forest_Area'] = mask_in[mask,2].sum()*convert_area 
        ##print('i' + str(ii))        
        Image_Features.loc[ind, 'Local_Bare_Area'] = mask_in[mask,3].sum()*convert_area   
        if verbose:
            print(str(ii) + ' / ' + str(nbuildings))
    return Image_Features





  
def Image_Generator(gen_dataframe, feature_dict, rot = 360, batch_size = 5, image_color_mode = "rgb",
                    mask_color_mode = "grayscale", save_to_dir = None, zoom_range = 0.2, 
                    horizontal_flip = True, vertical_flip = True, fill_mode = 'reflect',
                    target_size = (512,512), seed = 1, normalize = False, image_col = 'Image', 
                   label_col = 'Label', adjust = True):
    '''
    Use data augmentation to return both an image and mask that are modified in the same way. 
    Images from the generator can be saved by setting save_to_dir to path name. 
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


### Loss functions: 

###
def Weighted_Cross_Entropy(y_true, y_pred, eps = 1e-10):
    """Calculate weighted cross entropy"""
    y_pred = tf.cast(y_pred, 'float64')
    y_true = tf.cast(y_true, 'float64')
    # deduce weights based on true pixel value
    class_weights = weights * y_true
    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = y_true*tf.math.log(y_pred + eps)
    ##print(unweighted_losses.dtype, weights.dtype)
    weighted_losses = unweighted_losses * class_weights
    # reduce the result to get your final loss
    loss = -tf.reduce_sum(weighted_losses)
    return loss

## Toy example of a loss function that doesn't use all entries of an image
def Dice_Coef_NA(y_true, y_pred, index, smooth = 1e-7):
    """Calculate dice coefficient, used in dice_coef_multilabel"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    ## remove entries that don't count for land cover regions
    if index != 0 & index !=1:
        mask_include = (y_true_f!=255)
        y_true_f = y_true_f[mask_include]
        y_pred_f = y_pred_f[mask_include]
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


###
def Dice_Coef(y_true, y_pred, smooth = 1e-7):
    """Calculate dice coefficient, used in dice_coef_multilabel"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    den = (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    val = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    print('int: ', str(intersection), '_den: ', str(den),  '_val', str(val))
    return val 

###
def Dice_Coef_Multilabel(y_true, y_pred, numLabels=len(feature_dict)):
    dice=0
    for index in range(numLabels):
        dice -= Dice_Coef(y_true[:,:,:,index], y_pred[:,:,:, index])
    return dice    

###
def Dice_Coef_Multilabel_NA(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice -= Dice_Coef_NA(y_true[:,:,:,index], y_pred[:,:,:, index], index)
    return dice    

## Before being fed into the CNN, each image is normalized so that its values 
## fall between -1 and 1. Normalization is not absolutely required, but, 
## is an easy way to increase the speed with which the model learns patterns
## in the dataset. 
def Normalize_Image(img):
    """Scale image"""
    img = img / 127.5
    img = img - 1.0 
    return (img)  


def Predict_Class_Probabilities(array, model, step = 100, normalize = False, 
                                filename = None, path = ''):
    """Applies model to image and outputs image with class probabilities in 
    channels dimension.
    
    Parameters:
    -array: image array
    -model: CNN model object
    -step: step size of prediction window; multiple pixel predictions are averaged
    -normalize: option to normalize image before applying CNN (required for VGG) 
    -filename: save location if not None
    -path: directory of save location
    """
    
    img_height, img_width = model.input_shape[1:3]
    num_class = model.output_shape[3]
    image_orig = array.astype('float')
    if normalize:
        image_orig = Normalize_Image(image_orig)
    ## Create "batch" dimension for compatibility with CNN    
    image_orig = np.expand_dims(image_orig, axis=0)
    shp = np.array(array.shape)
    shp[2] = num_class
    predicted_image = 0.0*np.zeros(shp).astype('float')
    count_image = 0.0*predicted_image
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
    ## Average predictions
    predicted_image = np.divide(predicted_image,count_image); 
    predicted_class = predicted_image.argmax(axis = 2)    
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
    
        
### Previously called Process_Masks
def Predict_Image_Contours(img, mask_full, feature_dict, filename = None, path = ''):
    """ Takes image and mask as input and returns image with mask contours 
        
    Parameters:
    -img: image array
    -mask_full: expanded, one-hot-encoded mask (h,w,c)
    -feature_dict: dictionary that associates feature number and name
    -filename: save location if not None
    -path: directory of save location
    
    """
    for ii in range(len(feature_dict)):
        Type = feature_dict[str(ii)] ## So first key is 1
        if Type=='modern_build':
            color_rgb  = (255,0,0)
        elif Type=='trad_build':
            color_rgb = (0,0,255)
        mask = mask_full[:,:,ii]
        mask = 255*mask.round().astype('uint8')
        mask = np.stack((mask,mask, mask),-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY);
        ret, thresh = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ##print('here')
        area_thresh =30 ## Depends on what the minimal building size desired is
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



def Save_Image_Crop(img, x, y, width, height, filename = None, path = 'Predictions'):
    """Crop image and save PNG
    
    Parameters: 
    -img: image
    -x,y: bottom left corner of crop rectangle
    -width, height: width and height of crop rectangle
    """
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



##########

## Deprecated??
def Summarize_Features(Features):
    """Summarize output from Count_Features function into dataframe"""
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
