import os
import imp
import tensorflow as tf
import math
import numpy as np
import itertools
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def find1(item1, tofind):
  for i in range(len(item1)):
    if(item1[i].name == tofind):
      return item1[i]
  return None

def filename(index):
  if(index < 10):
    return "00"+str(index)
  elif(index < 100):
    return "0"+str(index)
  else:
    return str(index)

os.environ['PYTHONPATH']='/env/python:/home/eric/waymo-od'
m=imp.find_module('waymo_open_dataset', ['/home/eric/waymo-od'])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2 as label_dataset

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
  for index, image in enumerate(frame.images):
    name = open_dataset.CameraName.Name.Name(image.name)

    path = "images/" + name
    if(os.path.exists(path) == False):
      try:
        os.mkdir(path, 0o755)
      except OSError:
        print ("Creation of the directory %s failed" % path)

    path = path + "/" + filename(i)+'.jpg'

    tmp1 = find1(frame.projected_lidar_labels, image.name).labels
    if(len(tmp1) == 0):
      continue
  
    from io import BytesIO
    image = Image.open(BytesIO(image.image))
    draw = ImageDraw.Draw(image)
    for index1 in range(len(tmp1)):
      tmp = tmp1[index1].box

      center_x = tmp.center_x
      center_y = tmp.center_y
      width = tmp.width
      length = tmp.length

      left = center_x - (length/2)
      top = center_y - (width/2)
      right = center_x + (length/2)
      bottom = center_y + (width/2)

      typeText = label_dataset.Label.Type.Name(tmp1[index1].type)
      draw.text((left, top-20), typeText, font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", size=20))
      draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=4, fill='red')
    
    image.save(path, format='JPEG')