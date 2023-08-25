from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, maximum, DepthwiseConv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPool2D, ZeroPadding2D
from tensorflow.keras.layers import add,concatenate
from tensorflow.keras.activations import relu, softmax, swish
import tensorflow as tf
import tensorflow
import numpy as np

################################################
# get_LCNN_featureID_1: None - ext = False, ext2 = True
# get_LCNN_featureID_2: Demo - ext = True
################################################
def get_LCNN_featureID_1_2(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    
        ## mel embedding
    if use_mel :
        
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)




        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])


    
        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)

        
        
    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        
        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)
        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



################################################
# get_LCNN_featureID_3: S1 and S2 + murmurs
################################################
def get_LCNN_featureID_3(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_s1s2_input_shape, mel_mm_input_shape, use_s1s2 = True,use_mm = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
    mel1_mm = keras.Input(shape=(mel_mm_input_shape), name = 'mel_mm')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)




        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)

        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)
        
       
   # mel1_s1s2
    if use_mm :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_s1s2, mel1_mm] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

################################################
# get_LCNN_featureID_4: S1 and S2
################################################

def get_LCNN_featureID_4(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_s1s2_input_shape, use_s1s2 = True, use_mm = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)

        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)

        batch19 = batch19 + u
        ############################

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_s1s2])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_s1s2])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_s1s2] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


################################################
# get_LCNN_featureID_5: murmurs
################################################
def get_LCNN_featureID_5(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_mm_input_shape, use_s1s2 = True,use_mm = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    
#     mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
    mel1_mm = keras.Input(shape=(mel_mm_input_shape), name = 'mel_mm')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)

        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)




        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)
        
       
   # mel1_s1s2
    if use_mm :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_mm] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


################################################
# get_LCNN_featureID_6: envelope
################################################
def get_LCNN_featureID_6(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_envel_input_shape, use_s1s2 = True, use_mm = True, use_envel = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    mel1_envel = keras.Input(shape=(mel_envel_input_shape), name = 'mel_envel')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
    

    if use_envel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_envel)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_envel)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_envel = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)
        
       
   # mel1_s1s2
    if use_mm :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ### only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_envel])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_envel])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_envel] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


################################################
# get_LCNN_featureID_7 : s(S1 and S2) + s(murmur)
################################################
def get_LCNN_featureID_7(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_s1s2_input_shape, mel_mm_input_shape, use_s1s2 = True,use_mm = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
    mel1_mm = keras.Input(shape=(mel_mm_input_shape), name = 'mel_mm')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)




        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)
        
       
   # mel1_s1s2
    if use_mm :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)
        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_s1s2, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_s1s2, mel1_mm] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


#################################################
# get_LCNN_featureID_8 : s(S1 and S2)
#################################################
def get_LCNN_featureID_8(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_s1s2_input_shape, use_s1s2 = True, use_mm = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)




        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_s1s2])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_s1s2])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_s1s2] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

################################################
# get_LCNN_featureID_9: s(murmurs)
################################################

def get_LCNN_featureID_9(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_mm_input_shape, use_s1s2 = True,use_mm = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    
#     mel1_s1s2 = keras.Input(shape=(mel_s1s2_input_shape), name = 'mel_s1s2')
    mel1_mm = keras.Input(shape=(mel_mm_input_shape), name = 'mel_mm')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)




        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)
        
       
   # mel1_s1s2
    if use_mm :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_mm])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_mm] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

################################################
# get_LCNN_featureID_10: s(envelope)
################################################

def get_LCNN_featureID_10(mel_input_shape, cqt_input_shape, stft_input_shape, 
                       mel_envel_input_shape, use_s1s2 = True,use_mm = True,use_envel = True,
                       use_mel = True, use_cqt = True, use_stft = True, 
                       ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')

    mel1_envel = keras.Input(shape=(mel_envel_input_shape), name = 'mel_envel')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)

        batch19 = batch19 + u
        ############################

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2 = layers.GlobalAveragePooling2D()(mha)
        
    

    if use_envel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_envel)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_envel)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)
        


        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)




        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_envel = layers.GlobalAveragePooling2D()(mha)
        
   # mel1_s1s2
    if use_s1s2 :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_s1s2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)

        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_s1s2 = layers.GlobalAveragePooling2D()(mha)
        
       
   # mel1_s1s2
    if use_mm :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1_mm)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        max3 = tf.keras.activations.swish(max3)


        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)


        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        batch10 = tf.keras.activations.swish(batch10)

        ############################
        u = AveragePooling2D(pool_size=(max3.shape[1],1), strides=(2,2))(max3)
        u = Conv2D(filters=48, kernel_size=3, padding='same', activation=None)(u)

        batch10 = batch10 + u
        ############################

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        batch13 = tf.keras.activations.swish(batch13)


        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)
        max16 = tf.keras.activations.swish(max16)


        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        batch19 = tf.keras.activations.swish(batch19)


        ############################
        u = AveragePooling2D(pool_size=(batch10.shape[1],1), strides=(2,2))(batch10)
        u = Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(u)

        batch19 = batch19 + u
        ############################


        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)



        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 
        batch25 = tf.keras.activations.swish(batch25)


        ############################
        u = AveragePooling2D(pool_size=(batch19.shape[1],1), strides=(1,1))(batch19)
        u = Conv2D(filters=32, kernel_size=1, padding='same', activation=None)(u)
        
        batch25 = batch25 + u
        ############################

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        mha = layers.MultiHeadAttention(num_heads=8, key_dim=256)(mfm27,mfm27,mfm27)
        mel2_mm = layers.GlobalAveragePooling2D()(mha)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## only stft 
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ## only mel
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ## only cqt
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1, mel2_envel])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([rr1,mel2_envel])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1, mel1_envel] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)
