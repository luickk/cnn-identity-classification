from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

class inception_net:

    def __init__(self):
        print('Initializing CNN')

    def load_model(self,
                image_dir,
                summaries_dir,
                bottleneck_dir,
                final_result,
                final_tensor_name,
                testing_percentage,
                validation_percentage,
                learning_rate,
                how_many_training_steps,
                train_batch_size,
                eval_step_interval,
                validation_batch_size,
                output_graph,
                output_labels):


        self.image_dir = image_dir
        self.summaries_dir = summaries_dir
        self.bottleneck_dir = bottleneck_dir
        self.final_result = final_result
        self.final_tensor_name = final_tensor_name
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage
        self.learning_rate = learning_rate
        self.how_many_training_steps = how_many_training_steps
        self.train_batch_size = train_batch_size
        self.eval_step_interval = eval_step_interval
        self.validation_batch_size = validation_batch_size
        self.output_graph = output_graph
        self.output_labels = output_labels

        # Setup the directory we'll write summaries to for TensorBoard
        if tf.gfile.Exists(summaries_dir):
            tf.gfile.DeleteRecursively(summaries_dir)
        tf.gfile.MakeDirs(summaries_dir)

        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

        self.graph = graph
        self.bottleneck_tensor = bottleneck_tensor
        self.jpeg_data_tensor = jpeg_data_tensor
        self.resized_image_tensor = resized_image_tensor
        # Look at the folder structure, and create lists of all the images.
        image_lists = create_image_lists(image_dir, testing_percentage, validation_percentage)
        self.image_lists = image_lists
        print(image_lists)
        class_count = len(image_lists.keys())
        if class_count == 0:
          print('No valid folders of images found at ' + image_dir)
          return -1
        if class_count == 1:
          print('Only one valid folder of images found at ' + image_dir +
                ' - multiple classes are needed for classification.')
          return -1

        # See if the command-line self mean we're applying any distortions.
        do_distort_images = should_distort_images(False, 0, 0, 0)
        sess = tf.Session()
        self.sess = sess
        self.do_distort_images = do_distort_images
        if do_distort_images:
          # We will be applying distortions, so setup the operations we'll need.
          distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(False, 0, 0, 0)
        else:
          # We'll make sure we've calculated the 'bottleneck' image summaries and
          # cached them on disk.
          cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)

        # Add the new layer that we'll be training.
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(len(image_lists.keys()), final_tensor_name, bottleneck_tensor, learning_rate)
        self.train_step = train_step
        self.cross_entropy = cross_entropy
        self.bottleneck_input = bottleneck_input
        self.ground_truth_input = ground_truth_input
        self.final_tensor = final_tensor

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)
        self.evaluation_step =evaluation_step
        self.prediction = prediction
        # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
        merged = tf.summary.merge_all()
        self.merged = merged
        train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                             sess.graph)
        validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')
        self.train_writer = train_writer
        self.validation_writer = validation_writer
        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        return output_graph, output_labels

    def retrain(self):
        # Run the training for as many cycles as requested on the command line.
        for i in range(self.how_many_training_steps):
          # Get a batch of input bottleneck values, either calculated fresh every time
          # with distortions applied, or from the cache stored on disk.
          if self.do_distort_images:
            train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                sess, image_lists, train_batch_size, 'training',
                image_dir, distorted_jpeg_data_tensor,
                distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
          else:
            train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(self.sess, self.image_lists, self.train_batch_size, 'training', self.bottleneck_dir, self.image_dir, self.jpeg_data_tensor, self.bottleneck_tensor)
          # Feed the bottlenecks and ground truth into the graph, and run a training
          # step. Capture training summaries for TensorBoard with the `merged` op.
          train_summary, _ = self.sess.run([self.merged, self.train_step],
                   feed_dict={self.bottleneck_input: train_bottlenecks,
                              self.ground_truth_input: train_ground_truth})
          self.train_writer.add_summary(train_summary, i)

          # Every so often, print out how well the graph is training.
          is_last_step = (i + 1 == 4000)
          if (i % self.eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = self.sess.run(
                [self.evaluation_step, self.cross_entropy],
                feed_dict={self.bottleneck_input: train_bottlenecks,
                           self.ground_truth_input: train_ground_truth})
            print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                            train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                       cross_entropy_value))
            validation_bottlenecks, validation_ground_truth, _ = (
                get_random_cached_bottlenecks(
                    self.sess, self.image_lists, self.validation_batch_size, 'validation',
                    self.bottleneck_dir, self.image_dir, self.jpeg_data_tensor,
                    self.bottleneck_tensor))
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy = self.sess.run(
                [self.merged, self.evaluation_step],
                feed_dict={self.bottleneck_input: validation_bottlenecks,
                           self.ground_truth_input: validation_ground_truth})
            self.validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                  (datetime.now(), i, validation_accuracy * 100,
                   len(validation_bottlenecks)))

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(self.sess, self.image_lists, -1,
                                          'testing', self.bottleneck_dir, self.image_dir, self.jpeg_data_tensor, self.bottleneck_tensor))
        test_accuracy, predictions = self.sess.run(
            [self.evaluation_step, self.prediction],
            feed_dict={self.bottleneck_input: test_bottlenecks,
                       self.ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' % (
            test_accuracy * 100, len(test_bottlenecks)))

        # Write out the trained graph and labels with the weights stored as constants.
        output_graph_def = graph_util.convert_variables_to_constants(
            self.sess, self.graph.as_graph_def(), [self.final_tensor_name])
        with gfile.FastGFile(self.output_graph, 'wb') as f:
          f.write(output_graph_def.SerializeToString())
        with gfile.FastGFile(self.output_labels, 'w') as f:
          f.write('\n'.join(self.image_lists.keys()) + '\n')



def create_image_lists(image_dir, testing_percentage, validation_percentage):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
      if is_root_dir:
        is_root_dir = False
        continue
      extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png']
      file_list = []
      dir_name = os.path.basename(sub_dir)
      if dir_name == image_dir:
        continue
      print("Looking for images in '" + dir_name + "'")
      for extension in extensions:
        file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
        file_list.extend(gfile.Glob(file_glob))
      if not file_list:
        print('No files found')
        continue
      if len(file_list) < 20:
        print('WARNING: Folder has less than 20 images, which may cause issues.')
      elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
        print('WARNING: Folder {} has more than {} images. Some images will '
              'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
      label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
      training_images = []
      testing_images = []
      validation_images = []
      for file_name in file_list:
        base_name = os.path.basename(file_name)

        hash_name = re.sub(r'_nohash_.*$', '', file_name)

        hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                           (100.0 / MAX_NUM_IMAGES_PER_CLASS))
        if percentage_hash < validation_percentage:
          validation_images.append(base_name)
        elif percentage_hash < (testing_percentage + validation_percentage):
          testing_images.append(base_name)
        else:
          training_images.append(base_name)
      result[label_name] = {
          'dir': dir_name,
          'training': training_images,
          'testing': testing_images,
          'validation': validation_images,
      }
    return result



def create_inception_graph():
    with tf.Session() as sess:
        model_filename = os.path.join('cnn_tens/inception_v3/classify_image_graph_def.pb')
        print(model_filename)
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor
def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(sess, image_lists, label_name, index,
                                 image_dir, category, bottleneck_dir,
                                 jpeg_data_tensor, bottleneck_tensor)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          print(str(how_many_bottlenecks) + ' bottleneck files created.')
def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except:
    print("Invalid float found, recreating bottleneck")
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
  return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'


def get_image_path(image_lists, label_name, index, image_dir, category):
  if label_name not in image_lists:
      tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
      tf.logging.fatal('Category does not exist %s.', category)

  category_list = label_lists[category]
  if not category_list:
      tf.logging.fatal('Label %s has no images in the category %s.', label_name, category)
  print('asdasdddddddddddddddddddddddddd',index)
  print('fffffffffffffffffffffffffffffff', category_list)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path

def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor):
  print('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index, image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values



def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, learning_rate):
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)
def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def add_evaluation_step(result_tensor, ground_truth_tensor):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction

def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir, jpeg_data_tensor, bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
      bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                            image_index, image_dir, category,
                                            bottleneck_dir, jpeg_data_tensor,
                                            bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                              image_index, image_dir, category,
                                              bottleneck_dir, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames
