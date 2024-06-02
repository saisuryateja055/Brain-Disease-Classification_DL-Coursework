from pathlib import Path
import os
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def structure_datasets(base_dir="/content/Brain-Disease-Classification/datasets"):
  for dataset in os.listdir(base_dir): 
    ds_path = os.path.join(base_dir, dataset)
    print(ds_path)
    conts = os.listdir(ds_path)
    if ("train" in conts) and ("test" in conts):
      continue
    train_dir = os.path.join(ds_path, "train")
    test_dir = os.path.join(ds_path, "test")
    valid_dir = os.path.join(ds_path, "valid")
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(valid_dir)

    for class_name in conts:
      # img_paths = os.path.join(ds_path, os.listdir(os.path.join(ds_path, class_name)))
      img_paths = [os.path.join(ds_path, class_name, i) for i in os.listdir(os.path.join(ds_path, class_name))]
      labels = [class_name for i in range(len(img_paths))]
      if len(img_paths)==0:
        continue
      print(img_paths)
      x_train, x_temp, _, _ = train_test_split(img_paths, labels, random_state=43, test_size=0.4)
      x_test, x_valid, _, _ = train_test_split(x_temp, [class_name for i in range(len(x_temp))], random_state=43, test_size=0.4)
      print(len(x_train), len(x_test), len(x_valid))

      os.mkdir(os.path.join(train_dir, class_name))
      os.mkdir(os.path.join(test_dir, class_name))
      os.mkdir(os.path.join(valid_dir, class_name))

      for img in x_train:
        shutil.copy(img, os.path.join(train_dir, class_name))

      for img in x_test:
        shutil.copy(img, os.path.join(test_dir, class_name))

      for img in x_valid:
        shutil.copy(img, os.path.join(valid_dir, class_name))
      shutil.rmtree(os.path.join(ds_path, class_name))

 


def process(ds, batch_size, img_size, mode=1):
  h, w = img_size[0], img_size[1]

  resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(h, w),
    layers.Rescaling(1./255)
  ])

  augment = tf.keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical"),
      layers.RandomRotation(0.2),
      layers.RandomZoom(.5, .2)
  ])

  if mode==2:
    combined = tf.keras.Sequential([
        resize_and_rescale,
        augment
    ])

    # ds = ds.map(lambda x, y: (resize_and_rescale(x), y))
    # ds = ds.map(lambda x,y : (augment(x), y))

    ds = ds.map(lambda x, y: (combined(x), y))
  elif mode==1:
    ds = ds.map(lambda x, y : (resize_and_rescale(x), y))

  ds = ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
  return ds

from tensorflow.keras.utils import image_dataset_from_directory

def get_ds_splits(ds_name, base_dir="/content/Brain-Disease-Classification/datasets"):
  IMAGE_SIZE = (224, 224)
  ds_path = os.path.join(base_dir, ds_name)

  conts = os.listdir(ds_path)

  if ("train" not in conts) or ("test" not in conts):
    return "ERROR: Splits not detected"
  train_dir = os.path.join(ds_path, "train")
  test_dir = os.path.join(ds_path, "test")


  train_ds = image_dataset_from_directory(
      train_dir,
      image_size = IMAGE_SIZE,
      shuffle=True,
      seed = 21
  )
  train_ds = process(train_ds, 32, IMAGE_SIZE, 2)


  test_ds = image_dataset_from_directory(
      test_dir,
      image_size = IMAGE_SIZE,
      batch_size = 32,
      shuffle=True,
      seed = 21
  )

  test_ds = process(test_ds, 32, IMAGE_SIZE, 1)

  if "valid" in conts:
    valid_dir = os.path.join(ds_path, "valid")
    valid_ds = image_dataset_from_directory(
          valid_dir,
          image_size = IMAGE_SIZE,
          batch_size = 32,
          seed = 21
      )
    valid_ds = process(valid_ds, 32, IMAGE_SIZE, 1)
    return train_ds, test_ds, valid_ds
  return train_ds, test_ds, None

# train_ds, test_ds, valid_ds = get_ds_splits("Brain Stroke")



def visualize(train_ds, test_ds, valid_ds=None):
  fig = plt.figure(figsize=(10, 10))
  for img, label in train_ds.take(1):
    # print(img, label)
    # plt.subplot(1, 3, 1)
    ax1 = fig.add_subplot(131)
    ax1.title.set_text("TRAIN")
    plt.imshow(img[0, :, :, :])
    break

  for img, label in test_ds.take(1):
    # print(img, label)
    ax1 = fig.add_subplot(132)
    ax1.title.set_text("TEST")
    plt.imshow(img[0, :, :, :])
    break

  if valid_ds:
    for img, label in valid_ds.take(1):
      # print(img, label)
      ax1 = fig.add_subplot(133)
      ax1.title.set_text("VALID")
      plt.imshow(img[0, :, :, :])
      break
# visualize(train_ds, test_ds, valid_ds)
