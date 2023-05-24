# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 23:27:38 2023

@author: Eason
"""
import scipy.io
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input
from tensorflow.keras.layers import BatchNormalization, UpSampling2D

import numpy as np
import matplotlib.pyplot as plt


def combined_loss(y_true, y_pred, alpha=0.88):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    return alpha * mse_loss + (1 - alpha) * ssim_loss

def build_generator(input_shape):
    inputs = Input(input_shape)
    leaky_relu = tf.keras.layers.LeakyReLU()

    # Encoder (contraction) path
    c1 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(inputs)
    c1 = BatchNormalization()(c1)
    c1 = leaky_relu(c1)
    c1 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(c1)
    c1 = BatchNormalization()(c1)
    c1 = leaky_relu(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(p1)
    c2 = BatchNormalization()(c2)
    c2 = leaky_relu(c2)
    c2 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = leaky_relu(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(p2)
    c3 = BatchNormalization()(c3)
    c3 = leaky_relu(c3)
    c3 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(c3)
    c3 = BatchNormalization()(c3)
    c3 = leaky_relu(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottom layer
    c4 = Conv2D(512, (3, 3), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = leaky_relu(c4)
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = leaky_relu(c4)
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = leaky_relu(c4)

    # Decoder (expansion) path
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Conv2D(256, (2, 2), padding='same')(u5)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(u5)
    c5 = BatchNormalization()(c5)
    c5 = leaky_relu(c5)
    c5 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(c5)
    c5 = BatchNormalization()(c5)
    c5 = leaky_relu(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(128, (2, 2), padding='same')(u6)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(u6)
    c6 = BatchNormalization()(c6)
    c6 = leaky_relu(c6)
    c6 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(c6)
    c6 = BatchNormalization()(c6)
    c6 = leaky_relu(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(64, (2, 2), padding='same')(u7)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(u7)
    c7 = BatchNormalization()(c7)
    c7 = leaky_relu(c7)
    c7 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = leaky_relu(c7)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.009, beta_1=0.6, clipnorm=0.01, epsilon=0.001), loss=combined_loss, metrics=['accuracy'])
    model.summary()
    return model

def corrupt(image_data, corruption_level=0.1):
    # Flatten the image into a 1D array
    flattened = image_data.flatten()
    
    # Compute the number of pixels to set to 0
    num_corruptions = int(corruption_level * len(flattened))
    
    # Choose random pixel indices to set to 0
    indices = np.random.choice(len(flattened), size=num_corruptions, replace=False)
    
    # Set the chosen pixels to 0
    flattened[indices] = 0
    
    # Reshape the flattened array back into the original image shape
    corrupted = flattened.reshape(image_data.shape)
    
    return corrupted

def add_noise(image_data,corruption_level,arg_factor):
    complete_images = []
    corrupt_images = []
    for i in range(len(image_data)):
        #use training data to do argumention
        for j in range(arg_factor):
            corrupt_image = corrupt(image_data[i,:,:], corruption_level=corruption_level)
            corrupt_images.append(corrupt_image)
            complete_images.append(image_data[i,:,:])
            
    corrupt_images = np.array(corrupt_images)
    complete_images = np.array(complete_images)
    #reshpae (183,152,152) to (183,152,152,1)
    complete_images = np.expand_dims(complete_images, axis=-1)
    corrupt_images = np.expand_dims(corrupt_images, axis=-1)
    
    return corrupt_images,complete_images

if __name__ == "__main__":
    #load .mat file
    file_name  = "C:/Users/88696/Desktop/三下課程/影像處理/project3/Nevada.mat"
    mat_data = scipy.io.loadmat(file_name)
    hyperspectral_data = mat_data['X']
    
    length , width ,height = hyperspectral_data.shape
    target_size = (length,width)
    
    #initailize the model
    model = build_generator((152,152,1))
    
    images = [];
    for i in range(height):
        img = hyperspectral_data[:,:,i]
        img = np.array(img)
        # Pad the image with 1 pixel on each side make (150,150) => (152,152)
        img = np.pad(img, pad_width=1, mode='constant', constant_values=0)

        images.append(img)
    images = np.array(images)
    
    # Set the validation ratio, meaning the ratio of the data will be used for testing
    val_ratio = 0.92
    
    #choose which channel to be test data
    test_indices = np.random.choice(height, int(height*val_ratio), replace=False)
    # Create an array of all indices
    all_indices = np.arange(height)
    # Remove test indices to get train indices
    train_indices = np.delete(all_indices, test_indices)
    
    test_img = images[test_indices,:,:]
    train_img = images[train_indices,:,:]
    
    train_corrupt , train_complete = add_noise(train_img,0.97,60)
    test_corrupt , test_complete = add_noise(test_img,0.97,1)
    
    #train model
    model.fit(train_corrupt, train_complete, epochs=15, batch_size=16, verbose=1, validation_data=(test_corrupt, test_complete))
    
    
    pred = model.predict(test_corrupt)
    total_rmse = 0
    for j in range(len(test_corrupt)):
        plt.figure(figsize=(12, 4))

        pic_1 = np.reshape(test_corrupt[j,:,:,:],(152,152))
        pic_1 = (pic_1*255).astype(np.uint8)
        plt.subplot(1, 3, 1)
        plt.imshow(pic_1,cmap = 'gray')
        plt.title('corrupt')
        
        pic_2 = np.reshape(pred[j,:,:,:],(152,152))
        pic_2 = (pic_2*255).astype(np.uint8)
        plt.subplot(1, 3, 2)
        plt.imshow(pic_2,cmap = 'gray')
        plt.title('predict')

        
        pic_3 = np.reshape(test_complete[j,:,:,:],(152,152))
        pic_3 = (pic_3*255).astype(np.uint8)
        plt.subplot(1, 3, 3)
        plt.imshow(pic_3,cmap = 'gray')
        plt.title('ground truth')
        plt.show()
        
        difference = pred[j,:,:,:] - test_complete[j,:,:,:]
        squared_difference = np.square(difference)
        mean_squared_difference = np.mean(squared_difference)
        rmse = np.sqrt(mean_squared_difference)
        total_rmse += rmse
        
        print("RMSE: ",rmse)
    
    print("Avg rmse: ",total_rmse/np.shape(pred)[0])
            
    
