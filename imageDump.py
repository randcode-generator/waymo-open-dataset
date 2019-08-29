import os
import imp
import tensorflow as tf
import math
import numpy as np
import itertools

os.environ['PYTHONPATH']='/env/python:/home/eric/waymo-od'
m=imp.find_module('waymo_open_dataset', ['/home/eric/waymo-od'])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

tf.enable_eager_execution()

FILENAME = '/home/eric/waymo-od/tutorial/frames'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
if(os.path.exists("images") == False):
  try:
    os.mkdir("images", 0o755)
  except OSError:
    print ("Creation of the directory %s failed" % path)

i = 0
for data in dataset:
  frame = open_dataset.Frame()
  frame.ParseFromString(bytearray(data.numpy()))
  i += 1
  with tf.compat.v1.Session() as sess:
      for index, image in enumerate(frame.images):
        name = open_dataset.CameraName.Name.Name(image.name)
        path = "images/" + name
        if(os.path.exists(path) == False):
          try:
            os.mkdir(path, 0o755)
          except OSError:
            print ("Creation of the directory %s failed" % path)

        fname = tf.constant(path + "/" + str(i)+str(index)+'.jpg')
        print(path + "/" + str(i)+str(index)+'.jpg')
        fwrite = tf.io.write_file(fname, image.image)
        sess.run(fwrite)
          
