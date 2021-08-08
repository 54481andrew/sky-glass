def vgg_unet(pretrained_weights = None, input_size = (512,512,3), 
             fix_weights = False, dropout = 0, batch_norm = True, lr = 1e-3, 
             loss = 'dice', print_summary = False, num_class = 4):
 

    #Create your own input format 
    input = Input(shape=input_size,name = 'image_input')

    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_size, 
                            input_tensor=input)
    ##model_vgg16_conv.trainable = False  
    if fix_weights:
        for layer in model_vgg16_conv.layers:
            layer.trainable = False
    
    
    block1_conv2 = model_vgg16_conv.get_layer('block1_conv2').output
    block2_conv2 = model_vgg16_conv.get_layer('block2_conv2').output    
    block3_conv3 = model_vgg16_conv.get_layer('block3_conv3').output
    block4_conv3 = model_vgg16_conv.get_layer('block4_conv3').output
    block5_conv3 = model_vgg16_conv.get_layer('block5_conv3').output
    output_vgg16_conv = model_vgg16_conv.get_layer('block5_pool').output    
##15x15    
    
    x = Dropout(rate = dropout)(output_vgg16_conv)
    
    up7 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x))  
    if batch_norm: up7 = BatchNormalization()(up7) #NORM
    merge7 = concatenate([block5_conv3, up7], axis = 3)        
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge7)
    if batch_norm: conv7 = BatchNormalization()(conv7) #NORM    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(conv7)
    if batch_norm: conv7 = BatchNormalization()(conv7) #NORM


##30x30
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    if batch_norm: up8 = BatchNormalization()(up8) #NORM
    merge8 = concatenate([block4_conv3, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge8)
    if batch_norm: conv8 = BatchNormalization()(conv8) #NORM
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv8)
    if batch_norm: conv8 = BatchNormalization()(conv8) #NORM
##60x60
    
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    if batch_norm: up9 = BatchNormalization()(up9) #NORM
    merge9 = concatenate([block3_conv3,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge9)
    if batch_norm: conv9 = BatchNormalization()(conv9) #NORM    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv9)
    if batch_norm: conv9 = BatchNormalization()(conv9) #NORM
    
##56x56    

    up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    if batch_norm: up10 = BatchNormalization()(up10) #NORM
    merge10 = concatenate([block2_conv2,up10], axis = 3)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge10)
    if batch_norm: conv10 = BatchNormalization()(conv10) #NORM  
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv10)
    if batch_norm: conv10 = BatchNormalization()(conv10) #NORM  
##112x112    
        
    up11 = Conv2D(16, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
    if batch_norm: up11 = BatchNormalization()(up11) #NORM
    merge11 = concatenate([block1_conv2,up11], axis = 3)
    conv11 = Conv2D(16, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(merge11)
    if batch_norm: conv11 = BatchNormalization()(conv11) #NORM  
    conv11 = Conv2D(16, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv11)
    if batch_norm: conv11 = BatchNormalization()(conv11) #NORM  
##224x224
    
    
    conv12 = Conv2D(num_class, 1, activation = 'softmax')(conv11)

    model = Model(inputs = input, outputs = conv12)
    
    if loss=='dice':
        model.compile(optimizer = Adam(lr = lr), loss = Dice_Coef_Multilabel, metrics = [tf.metrics.categorical_accuracy, Dice_Coef_Multilabel, tf.metrics.categorical_crossentropy])
    elif loss == 'cross':
        model.compile(optimizer = Adam(lr = lr), loss = weighted_ce, metrics = [tf.metrics.categorical_accuracy, Dice_Coef_Multilabel,tf.metrics.categorical_crossentropy])
    else: 
        print('Need a valid loss option!!')
    
    if print_summary:
        model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model