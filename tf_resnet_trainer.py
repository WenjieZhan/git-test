"Trainer for amsoftmax loss"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from utils.losses import AMSoftmaxLoss
from utils.goods_utils import train_transform, dev_test_transform, \
    goods_loader_train_new, goods_loader_test_new, DatasetsFromList_Path, DatasetsFromList
from config.goods_config import *
import torch
# from network.face_resnet import ResnetFace50
from network.tf_resnet import ResNet50
from train.train_fn import save_checkpoint, test_cls
from utils.metrics import AccumulatedAccuracyMetric
import time
import numpy as np
import tensorflow as tf
import keras


def run_training(n_classes):
    data_dir = 'C:/Users/wk/Desktop/bky/dataSet/'
    image, label = get_files(data_dir)
    image_batches, label_batches = get_batches(image, label, 32, 32, 16, 20)
    model = ResNet50(n_classes)
    p = model.call(image_batches)
    cost = loss(p, label_batches)
    train_op = training(cost, 0.001)
    acc = get_accuracy(p, label_batches)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(1000):
            print(step)
            if coord.should_stop():
                break
            _, train_acc, train_loss = sess.run([train_op, acc, cost])
            print("loss:{} accuracy:{}".format(train_loss, train_acc))
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            class_train.append(filename+train_class+'/'+pic)
            label_train.append(train_class)
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    #shuffle the samples
    np.random.shuffle(temp)
    #after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    #print(label_list)
    return image_list,label_list


def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c, channels=3)
    # resize
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    # (x - mean) / adjusted_stddev
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    return images_batch, labels_batch


def loss(logits,label_batches):
     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
     cost = tf.reduce_mean(cross_entropy)
     return cost

def get_accuracy(logits,labels):
     acc = tf.nn.in_top_k(logits,labels,1)
     acc = tf.cast(acc,tf.float32)
     acc = tf.reduce_mean(acc)
     return acc

def training(loss,lr):
     train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
     return train_op

if __name__ == '__main__':

    train_dataset = DatasetsFromList_Path(args.data_dir, args.run_name, 'train', loader=goods_loader_train_new,
                                          transform=train_transform)
    n_classes = len(train_dataset.classes)
    run_training(n_classes)