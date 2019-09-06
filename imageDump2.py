import os
import imp
import tensorflow as tf
import math
import numpy as np
import itertools
from PIL import Image
from PIL import ImageDraw

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

tf.enable_eager_execution()

FILENAME = '/home/eric/waymo-od/tutorial/frames'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
if(os.path.exists("images") == False):
  try:
    os.mkdir("images", 0o755)
  except OSError:
    print ("Creation of the directory %s failed" % path)

def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index = 0):
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  lasers = sorted(frame.lasers, key=lambda laser: laser.name)
  points = [] 
  cp_points = []
  
  frame_pose = tf.convert_to_tensor(
      np.reshape(np.array(frame.pose.transform), [4, 4]))
  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == open_dataset.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.where(range_image_mask))

    cp = camera_projections[c.name][0]
    cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())

  return points, cp_points

def parse_range_image_and_camera_projection(frame):
  range_images = {}
  camera_projections = {}
  range_image_top_pose = None
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:
      range_image_str_tensor = tf.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == open_dataset.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = open_dataset.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

      camera_projection_str_tensor = tf.decode_compressed(
          laser.ri_return1.camera_projection_compressed, 'ZLIB')
      cp = open_dataset.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name] = [cp]
    if len(laser.ri_return2.range_image_compressed) > 0:
      range_image_str_tensor = tf.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

      camera_projection_str_tensor = tf.decode_compressed(
          laser.ri_return2.camera_projection_compressed, 'ZLIB')
      cp = open_dataset.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name].append(cp)
  return range_images, camera_projections, range_image_top_pose 

i = 0
for data in dataset:
  frame = open_dataset.Frame()
  frame.ParseFromString(bytearray(data.numpy()))
  (range_images, camera_projections,
    range_image_top_pose) = parse_range_image_and_camera_projection(frame)
  points, cp_points = convert_range_image_to_point_cloud(
      frame,
      range_images,
      camera_projections,
      range_image_top_pose)
  points_ri2, cp_points_ri2 = convert_range_image_to_point_cloud(
      frame,
      range_images,
      camera_projections,
      range_image_top_pose,
      ri_index=1)
  # 3d points in vehicle frame.
  points_all = np.concatenate(points, axis=0)
  points_all_ri2 = np.concatenate(points_ri2, axis=0)
  
  # camera projection corresponding to each point.
  cp_points_all = np.concatenate(cp_points, axis=0)
  cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

  cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
  cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

  i += 1

  #images = sorted(frame.images, key=lambda i:i.name)
  for index, image in enumerate(frame.images):
    name = open_dataset.CameraName.Name.Name(image.name)
    path = "images/" + name
    if(os.path.exists(path) == False):
      try:
        os.mkdir(path, 0o755)
      except OSError:
        print ("Creation of the directory %s failed" % path)

    path = path + "/" + filename(i)+'.jpg'

    with tf.compat.v1.Session() as sess:
      fname = tf.constant(path)
      print(path)
      fwrite = tf.io.write_file(fname, image.image)
      sess.run(fwrite)

    #here
    import matplotlib.pyplot as plt

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
    
    mask = tf.equal(cp_points_all_tensor[..., 0], image.name)

    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points = tf.concat(
      [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

    name = open_dataset.CameraName.Name.Name(image.name)
    path = "images/" + name
    path = path + "/" + filename(i)+'.jpg'

    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    offset = 1
    for point in projected_points:
      x_center = point[0]
      y_center = point[1]
      color = point[2]
      c = plt.get_cmap('jet')((color % 20.0) / 20.0)
      left = x_center - offset
      top = y_center - offset
      right = x_center + offset
      bottom = y_center + offset
      r = c[0]*255
      g = c[1]*255
      b = c[0]*255
      draw.ellipse((left,top,right,bottom), fill=(int(r), int(g), int(b), 128))
    image.save(path, format='JPEG')
