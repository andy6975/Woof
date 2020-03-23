import os
import warnings
import xml.etree.ElementTree as ET
import random
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# import tensorflow as tf

warnings.filterwarnings('ignore',category=FutureWarning)


anno_dir = '../Data_Woof/Annotation/'
imag_dir = '../Data_Woof/Images/'

classes = os.listdir(imag_dir)
images_npy, all_images = [], []
image_database = {}

num_classes = int(len(classes))
num_images = 0

for breed in classes:
    image_names = os.listdir(imag_dir + '/' + breed)
    num_images += len(image_names)
    image_database[breed] = image_names
    all_images += image_names

all_images = random.sample(all_images, num_images)
checkpoint = int(len(all_images) * 0.7)
train_data = all_images[:checkpoint]
test_data = all_images[checkpoint:]

def resize_image(img, size=(28,28)):
    c = 3
    h, w = img.shape[:2]

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    if h > w:
        dif = h
    else:
        dif = w

    if dif > (size[0]+size[1])//2:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def anno_reader(image_name, folder):
    anno_name = anno_dir + folder + '/' + image_name[:-4]
    doc = ET.parse(anno_name)
    size = doc.find('size')
    object = doc.find('object')
    bbox = object.find('bndbox')

    width = int(size.find('width').text)
    height = int(size.find('height').text)
        
    name = object.find('name').text

    x_min = int(bbox.find('xmin').text)
    y_min = int(bbox.find('ymin').text)
    x_max = int(bbox.find('xmax').text)
    y_max = int(bbox.find('ymax').text)

    # print(imag_dir + folder + '/' + image_name)

    img = cv2.imread(imag_dir + folder + '/' + image_name)
    img_crop = img[y_min:y_max, x_min:x_max]
    img_resize = resize_image(img_crop, (380, 480))

    cv2.imshow('Real', img)
    cv2.imshow('Resi', img_resize)
    cv2.imshow('BB', img_crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [x_min, y_min, x_max, y_max]

def batch_gen(batch_size=10):
    batch = []
    rand_indices = np.random.randint(0, len(train_data) - 1, batch_size)
    batch_images = [train_data[ind] for ind in rand_indices]
    for b_img in batch_images:
        for val in image_database.values():
            if b_img in val:
                folder = list(image_database.keys())[list(image_database.values()).index(val)]
                coordinates = anno_reader(b_img, folder)
                packet = [b_img, coordinates]
                batch.append(packet)

    print(batch)

batch_gen()




















# def class_process(class_name, class_id):
#     global count
#     annotations = os.listdir(anno_dir + class_name + '/')
#     for image_id in range(len(annotations)):
#         image_name = image_dir + class_name + '/' + annotations[image_id] + '.jpg'
#         anno_name = anno_dir + class_name + '/' + annotations[image_id]
#         doc = ET.parse(anno_name)
#         size = doc.find('size')
#         object = doc.find('object')
#         bbox = object.find('bndbox')

#         width = int(size.find('width').text)
#         height = int(size.find('height').text)
        
#         name = object.find('name').text

#         x_min = int(bbox.find('xmin').text)
#         y_min = int(bbox.find('ymin').text)
#         x_max = int(bbox.find('xmax').text)
#         y_max = int(bbox.find('ymax').text)

#         img = cv2.imread(image_name)
#         img_crop = img[y_min:y_max, x_min:x_max]
#         img_resize = resize_image(img_crop, (28, 28))

#         global images_npy
#         images_npy.append(img_resize)

#         global labels_npy
#         labels_npy[image_id + count][class_id] = 1.

#     #     cv2.imshow("real", img)
#     #     cv2.waitKey(0)
#     #     cv2.imshow("resized", img_resize)
#     #     cv2.waitKey(0)

#     # cv2.destroyAllWindows()
#     count += int(len(annotations))

# batch_size = 64
# for i in range(len(classes)):
#     print(i, classes[i])
#     class_process(classes[i], i)

# # images_npy = np.array(images_npy, dtype='uint8')
# # np.save("labels.npy", labels_npy)
# # np.save("images.npy", images_npy)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# # config.gpu_options.per_process_gpu_memory_fraction = 0.5
# global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.0002, global_step, 500, 0.95, staircase=True)
# saver = tf.train.Saver()

