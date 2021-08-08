def eff_unet(BVal = 0, pretrained_weights = None, input_size = (512,512,3), 
             fix_weights = False, dropout = 0, batch_norm = True, lr = 1e-3, 
             loss = 'dice', print_summary = False, num_class = 4):
 

    #Create your own input format 
    input = Input(shape=input_size,name = 'image_input')

    ## Choose the complexity of the EfficientNet backbone
    
    #B0
    if BVal==0:
        #Get back the convolutional part of a EfficientNet network trained on ImageNet
        model_eff_conv = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_size, 
                                input_tensor=input)
        ##model_vgg16_conv.trainable = False  
        if fix_weights:
            for layer in model_eff_conv.layers:
                layer.trainable = False
        block1 = model_eff_conv.get_layer('block1a_project_bn').output
        block2 = model_eff_conv.get_layer('block2b_add').output
        block3 = model_eff_conv.get_layer('block3b_add').output    
        block4 = model_eff_conv.get_layer('block4c_add').output
        block5 = model_eff_conv.get_layer('block5c_add').output
        block6 = model_eff_conv.get_layer('block6d_add').output
        block7 = model_eff_conv.get_layer('top_activation').output

    #B1
    if BVal==1:
        #Get back the convolutional part of a EfficientNet network trained on ImageNet
        model_eff_conv = EfficientNetB1(weights='imagenet', include_top=False, input_shape=input_size, 
                                input_tensor=input)
        ##model_vgg16_conv.trainable = False  
        if fix_weights:
            for layer in model_eff_conv.layers:
                layer.trainable = False
        block1 = model_eff_conv.get_layer('block1b_add').output
        block2 = model_eff_conv.get_layer('block2c_add').output
        block3 = model_eff_conv.get_layer('block3c_add').output    
        block4 = model_eff_conv.get_layer('block4d_add').output
        block5 = model_eff_conv.get_layer('block5d_add').output
        block6 = model_eff_conv.get_layer('block6d_add').output
        block7 = model_eff_conv.get_layer('top_activation').output   
    
    #B2
    if BVal==2:
        #Get back the convolutional part of a EfficientNet network trained on ImageNet
        model_eff_conv = EfficientNetB2(weights='imagenet', include_top=False, input_shape=input_size, 
                                input_tensor=input)
        ##model_vgg16_conv.trainable = False  
        if fix_weights:
            for layer in model_eff_conv.layers:
                layer.trainable = False
        block1 = model_eff_conv.get_layer('block1b_add').output
        block2 = model_eff_conv.get_layer('block2c_add').output
        block3 = model_eff_conv.get_layer('block3c_add').output    
        block4 = model_eff_conv.get_layer('block4d_add').output
        block5 = model_eff_conv.get_layer('block5d_add').output
        block6 = model_eff_conv.get_layer('block6e_add').output
        block7 = model_eff_conv.get_layer('top_activation').output
    
    #B3
    if BVal==3:
        #Get back the convolutional part of a EfficientNet network trained on ImageNet
        model_eff_conv = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_size, 
                                input_tensor=input)
        ##model_vgg16_conv.trainable = False  
        if fix_weights:
            for layer in model_eff_conv.layers:
                layer.trainable = False
        block1 = model_eff_conv.get_layer('block1b_add').output
        block2 = model_eff_conv.get_layer('block2c_add').output
        block3 = model_eff_conv.get_layer('block3c_add').output    
        block4 = model_eff_conv.get_layer('block4e_add').output
        block5 = model_eff_conv.get_layer('block5e_add').output
        block6 = model_eff_conv.get_layer('block6f_add').output
        block7 = model_eff_conv.get_layer('top_activation').output  


    ##B7
    if BVal==7:
        #Get back the convolutional part of a EfficientNet network trained on ImageNet
        model_eff_conv = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_size, 
                                input_tensor=input)
        ##model_vgg16_conv.trainable = False  
        if fix_weights:
            for layer in model_eff_conv.layers:
                layer.trainable = False
        block1 = model_eff_conv.get_layer('block1d_add').output
        block2 = model_eff_conv.get_layer('block2g_add').output
        block3 = model_eff_conv.get_layer('block3g_add').output    
        block4 = model_eff_conv.get_layer('block4j_add').output
        block5 = model_eff_conv.get_layer('block5j_add').output
        block6 = model_eff_conv.get_layer('block6m_add').output
        block7 = model_eff_conv.get_layer('top_activation').output
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(block7))
    if batch_norm: up8 = BatchNormalization()(up8) #NORM
    merge8 = concatenate([block5, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge8)
    if batch_norm: conv8 = BatchNormalization()(conv8) #NORM
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv8)
    if batch_norm: conv8 = BatchNormalization()(conv8) #NORM

    ##
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    if batch_norm: up9 = BatchNormalization()(up9) #NORM
    merge9 = concatenate([block3,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge9)
    if batch_norm: conv9 = BatchNormalization()(conv9) #NORM    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv9)
    if batch_norm: conv9 = BatchNormalization()(conv9) #NORM

    ##
    up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    if batch_norm: up10 = BatchNormalization()(up10) #NORM
    merge10 = concatenate([block2,up10], axis = 3)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge10)
    if batch_norm: conv10 = BatchNormalization()(conv10) #NORM  
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv10)
    if batch_norm: conv10 = BatchNormalization()(conv10) #NORM  

    ##
    up11 = Conv2D(16, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
    if batch_norm: up11 = BatchNormalization()(up11) #NORM
    merge11 = concatenate([block1,up11], axis = 3)
    conv11 = Conv2D(16, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge11)
    if batch_norm: conv11 = BatchNormalization()(conv11) #NORM  
    conv11 = Conv2D(16, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv11)
    if batch_norm: conv11 = BatchNormalization()(conv11) #NORM  

    ##
    up12 = Conv2D(16, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv11))
    if batch_norm: up12 = BatchNormalization()(up12) #NORM
    conv12 = Conv2D(16, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(up12)
    if batch_norm: conv12 = BatchNormalization()(conv12) #NORM  
    conv12 = Conv2D(16, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv12)
    if batch_norm: conv12 = BatchNormalization()(conv12) #NORM  



    conv13 = Conv2D(num_class, 1, activation = 'softmax')(conv12)
        
    model = Model(inputs = input, outputs = conv13)
    
    if loss=='dice':
        model.compile(optimizer = Adam(lr = lr), loss = Dice_Coef_Multilabel, metrics = [tf.metrics.categorical_accuracy, Dice_Coef_Multilabel, tf.metrics.categorical_crossentropy])
    elif loss == 'cross':
        model.compile(optimizer = Adam(lr = lr), loss = tf.metrics.categorical_crossentropy, metrics = [tf.metrics.categorical_accuracy, Dice_Coef_Multilabel, tf.metrics.categorical_crossentropy])
    elif loss == 'wcross':
        model.compile(optimizer = Adam(lr = lr), loss = Weighted_Cross_Entropy, 
                      metrics = [tf.metrics.categorical_accuracy, Dice_Coef_Multilabel, 
                                 tf.metrics.categorical_crossentropy])
    elif loss=='mse':
        model.compile(optimizer = Adam(lr = lr), loss = tf.metrics.mean_squared_error)        
    else: 
        print('Need a valid loss option!!')
    
    if print_summary:
        model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model