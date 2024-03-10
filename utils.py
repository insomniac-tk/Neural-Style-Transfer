import os
import numpy as np
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt



def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_img(path_to_img,
             max_dim = 512):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img



def remove_batch(img):
    if len(img.shape) > 3:
        img = tf.squeeze(img,axis=0)
    return img



def plot_images(content_image,
                style_image,
                result_image):
    content_image = remove_batch(content_image)
    style_image = remove_batch(style_image)
    result_image = remove_batch(result_image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Neural Style Transfer", fontsize=16)
    

    axes[0].imshow(content_image)
    axes[0].axis('off')  # Turn off axis
    axes[0].set_title('Content Image')


    axes[1].imshow(style_image)
    axes[1].axis('off')  
    axes[1].set_title('Style Image')

  
    axes[2].imshow(result_image)
    axes[2].axis('off')  
    axes[2].set_title('Result')
    plt.tight_layout()
    plt.show()


def validate_path(image_path):
   return os.path.exists(image_path)