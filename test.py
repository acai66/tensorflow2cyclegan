#from utils import *
import networks
import helpers
import time
import os
from tqdm import tqdm
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from instancenormalization import InstanceNormalization


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchSize', type=int, default=0, help='Batch Size to be used for training, 0 means transfer all test images')
    parser.add_argument('--data_dir', type=str, default='data/vangogh2photo/', help='Directory where train and test images are present')
    parser.add_argument('--out_dir', type=str, default='', help='Directory where output images will be put in')
    parser.add_argument('--temp', type=int, default=0, help='Num of temp weight restore from')

    opt, _ = parser.parse_known_args()

    data_dir = opt.data_dir
    out_dir = opt.out_dir if opt.out_dir else opt.data_dir
    batch_size = opt.batchSize
    temp = opt.temp

    print("Data_dir:{}".format(data_dir))
    print("BatchSize:{}".format(batch_size))


    if temp:
        genA2B = load_model(os.path.join('weight', 'generatorAToB_temp_%d.h5' % temp), custom_objects={'InstanceNormalization':InstanceNormalization})
        genB2A = load_model(os.path.join('weight', 'generatorBToA_temp_%d.h5' % temp), custom_objects={'InstanceNormalization':InstanceNormalization})
    else:
        genA2B = load_model(os.path.join('weight', 'generatorAToB.h5'), custom_objects={'InstanceNormalization':InstanceNormalization})
        genB2A = load_model(os.path.join('weight', 'generatorBToA.h5'), custom_objects={'InstanceNormalization':InstanceNormalization})

    if batch_size == 0:
        testA = os.listdir(os.path.join(data_dir, 'testA'))
        testB = os.listdir(os.path.join(data_dir, 'testB'))
        if not os.path.exists(os.path.join(out_dir, 'outputA')):
            os.mkdir(os.path.join(out_dir, 'outputA'))
        if not os.path.exists(os.path.join(out_dir, 'outputB')):
            os.mkdir(os.path.join(out_dir, 'outputB'))
        
        print('transfer %stestA to %soutputA' % (data_dir, out_dir))
        for A in tqdm(testA):
            img_path = os.path.join(os.path.join(data_dir, 'testA'), A)
            img = helpers.load_image_train(img_path)
            img = tf.expand_dims(img, 0)
            # Generate images
            fakeB = genA2B(img)
            fakeB = tf.squeeze(fakeB, 0)
            tf.keras.preprocessing.image.save_img(os.path.join(os.path.join(out_dir, 'outputA'), A), fakeB)
            
        print('transfer %stestB to %soutputB' % (data_dir, out_dir))
        for B in tqdm(testB):
            img_path = os.path.join(os.path.join(data_dir, 'testB'), B)
            img = helpers.load_image_train(img_path)
            img = tf.expand_dims(img, 0)
            # Generate images
            fakeA = genB2A(img)
            fakeA = tf.squeeze(fakeA, 0)
            tf.keras.preprocessing.image.save_img(os.path.join(os.path.join(out_dir, 'outputB'), B), fakeA)
    
        
    else:
        testA, testB = helpers.load_test_images(data_dir=data_dir, batch_size=batch_size)
    
        inputA = next(testA)
        inputB = next(testB)
        # Generate images
        fakeB = genA2B(inputA)
        fakeA = genB2A(inputB)

        # Get reconstructed images
        reconsA = genB2A(fakeB)
        reconsB = genA2B(fakeA)

        identityA = genB2A(inputA)
        identityB = genA2B(inputB)

        helpers.save_test_results(inputA, inputB, fakeA, fakeB, reconsA, reconsB, identityA, identityB)
        
    

    
