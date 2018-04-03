import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import trange
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

epochs = 15
batch_size = 20
keep_prob_ = 0.4
learning_rate_ = 1e-4


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph=tf.get_default_graph()

    return graph.get_tensor_by_name(vgg_input_tensor_name), \
        graph.get_tensor_by_name(vgg_keep_prob_tensor_name), \
        graph.get_tensor_by_name(vgg_layer3_out_tensor_name), \
        graph.get_tensor_by_name(vgg_layer4_out_tensor_name), \
        graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
    # l7 1x1 and deconv
    l7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, (1,1), (1,1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    l7_deconv = tf.layers.conv2d_transpose(l7_conv, num_classes, (4,4), (2,2), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # L4 1x1 conv, add l7 output and deconv
    l4_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, (1,1), (1,1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    l4_add = tf.add(l4_conv, l7_deconv)
    l4_deconv = tf.layers.conv2d_transpose(l4_add , num_classes, (4,4), (2,2), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # L3 1x1 conv, add l4 output and deconv
    l3_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, (1,1), (1,1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    l3_add = tf.add(l3_conv, l4_deconv)
    l3_deconv = tf.layers.conv2d_transpose(l3_add, num_classes, (16,16), (8,8), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return l3_deconv
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, 
        labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    loss = 1e6
    samples = list()
    losses = list()
    sample=0
    for epoch in trange(epochs):
        _count = 0
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                feed_dict={
                input_image: image,
                correct_label: label,
                keep_prob: keep_prob_,
                learning_rate: learning_rate_
                })
            samples.append(sample)
            losses.append(loss)
            print("#{:0>4} {:>10}: {:.2f}".format(_count,sample, loss))
            sample+=batch_size
            _count+=1
    plt.plot(samples, losses, 'b-')
    plt.savefig('./runs/loss.png')
    
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Parameters
    global epochs, batch_size, keep_prob, learning_rate_
    epochs = 25
    batch_size = 3
    keep_prob_ = 0.5
    learning_rate_ = 1e-4

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        labels=tf.placeholder(tf.int32)
        learning_rate=tf.placeholder(tf.float32)

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_layer, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        last_layer = layers(l3_out, l4_out, l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(last_layer, labels, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        saver = tf.train.Saver()

        train_nn(sess, epochs, batch_size, get_batches_fn, 
            train_op, cross_entropy_loss, input_layer,
            labels, keep_prob, learning_rate)

        model_save_path = runs_dir + \
            '/model_{}epochs_{}kp_{}batch.ckpt'.format(epochs, keep_prob_, batch_size)
        print("Saving models at: {}".format(model_save_path))
        saver.save(sess, model_save_path)
        # use helper function to save test data
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)    
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
