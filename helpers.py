import tensorflow as tf
import os.path
from tqdm import tqdm

cfg = {
        'height': 256,
        'width': 256
    }

def load_img(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image)
    image = tf.cast(image, tf.float32)
    return image

def normalize(image):
    image = (image / 127.5) - 1

    return image

def load_image_train(image_file):
    image = load_img(image_file)
    image = normalize(image)
    
    return image

def load_train_images(data_dir, batch_size):
    trainA_dataset = tf.data.Dataset.list_files(os.path.join(data_dir, os.path.join('trainA', '*.jpg')))
    trainB_dataset = tf.data.Dataset.list_files(os.path.join(data_dir, os.path.join('trainB', '*.jpg')))
    trainA_dataset = trainA_dataset.map(load_image_train).batch(batch_size, drop_remainder=True).repeat()
    trainB_dataset = trainB_dataset.map(load_image_train).batch(batch_size, drop_remainder=True).repeat()

    return iter(trainA_dataset), iter(trainB_dataset) 

def load_test_images(data_dir, batch_size):

    testA_dataset = tf.data.Dataset.list_files(os.path.join(data_dir, os.path.join('testA', '*.jpg')))
    testB_dataset = tf.data.Dataset.list_files(os.path.join(data_dir, os.path.join('testB', '*.jpg')))
    testA_dataset = testA_dataset.map(load_image_train).batch(batch_size, drop_remainder=True).repeat()
    testB_dataset = testB_dataset.map(load_image_train).batch(batch_size, drop_remainder=True).repeat()

    return iter(testA_dataset), iter(testB_dataset)

def save_test_results(testA, testB, fakeA, fakeB, reconsA, reconsB, identityA, identityB):
    for i in tqdm(range(testA.shape[0])):
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'realA_%d.png' % i), testA[i])
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'realB_%d.png' % i), testB[i])
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'fakeA_%d.png' % i), fakeA[i])
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'fakeB_%d.png' % i), fakeB[i])
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'reconsA_%d.png' % i), reconsA[i])
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'reconsB_%d.png' % i), reconsB[i])
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'identityA_%d.png' % i), identityA[i])
        tf.keras.preprocessing.image.save_img(os.path.join('results', 'identityB_%d.png' % i), identityB[i])



