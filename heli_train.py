from __future__ import print_function
import os
import tensorflow as tf
import heli_inference
import tf_utils
import numpy

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

TRAINING_NUMS=21000

MODEL_SAVE_PATH = "Model_Folder/"
MODEL_NAME = "model.ckpt"
# IMPORTS

# Keras' "get_session" function gives us easy access to the session where we train the graph
from keras import backend as K

# freeze_graph "screenshots" the graph
from tensorflow.python.tools import freeze_graph
# optimize_for_inference lib optimizes this frozen graph
from tensorflow.python.tools import optimize_for_inference_lib

# os and os.path are used to create the output file where we save our frozen graphs
import os.path as path

# EXPORT GAPH FOR UNITY
def export_model(saver, input_node_names, output_node_name):
    # creates the 'out' folder where our frozen graphs will be saved
    if not path.exists('out'):
        os.mkdir('out')

    # an arbitrary name for our graph
    GRAPH_NAME = 'heli_test_to _Unity'

    # GRAPH SAVING - '.pbtxt'
    tf.train.write_graph(K.get_session().graph_def, 'out', GRAPH_NAME + '_graph.pbtxt')

    # GRAPH SAVING - '.chkp'
    # KEY: This method saves the graph at it's last checkpoint (hence '.chkp')
    saver.save(K.get_session(), 'out/' + GRAPH_NAME + '.chkp')

    # GRAPH SAVING - '.bytes'
    # freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                           # input_binary, checkpoint_path, output_node_names,
                           # restore_op_name, filename_tensor_name,
                           # output_frozen_graph_name, clear_devices, "")
    freeze_graph.freeze_graph('out/' + GRAPH_NAME + '_graph.pbtxt', None, False,
                              'out/' + GRAPH_NAME + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + GRAPH_NAME + '.bytes', True, "")
    # freeze_graph.freeze_graph(input_graph='out/' + GRAPH_NAME + '_graph.pbtxt',
    #                           input_binary=True,
    #                           input_checkpoint='out/' + GRAPH_NAME + '.chkp',
    #                           output_node_names=output_node_name,
    #                           output_graph='out/frozen_' + GRAPH_NAME + '.bytes',
    #                           clear_devices=True, initializer_nodes="", input_saver="",
    #                           restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")
    # GRAPH OPTIMIZING
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + GRAPH_NAME + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + GRAPH_NAME + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

def next_batch(X_train,y_train,epochs_completed,index_in_epoch, batch_size, fake_data=False, shuffle=True):
    start = index_in_epoch  #index_in_epoch  所有的调用，总共用了多少个样本，相当于一个全局变量 #start第一个batch为0，剩下的就和index_in_epoch一样，如果超过了一个epoch，在下面还会重新赋值。
    # Shuffle for the first epoch 第一个epoch需要shuffle
    if epochs_completed == 0 and start == 0 and shuffle:
      # perm0 = numpy.arange(TRAINING_NUMS)  #生成的一个所有样本长度的np.array
      # numpy.random.shuffle(perm0)
      # X_train = X_train[perm0]
      # y_train = y_train[perm0]
      numpy.random.shuffle(X_train)
      numpy.random.shuffle(y_train)
    # Go to the next epoch


    if start + batch_size > TRAINING_NUMS: #epoch的结尾和下一个epoch的开头
      # Finished epoch
      epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = TRAINING_NUMS - start  # 最后不够一个batch还剩下几个
      images_rest_part = X_train[start:TRAINING_NUMS]
      labels_rest_part = y_train[start:TRAINING_NUMS]
      # Shuffle the data
      if shuffle:
          numpy.random.shuffle(X_train)
          numpy.random.shuffle(y_train)

      # Start next epoch
      start = 0
      index_in_epoch = batch_size - rest_num_examples
      end = index_in_epoch
      images_new_part = X_train[start:end]
      labels_new_part = y_train[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:  # 除了第一个epoch，以及每个epoch的开头，剩下中间batch的处理方式
      index_in_epoch += batch_size # start = index_in_epoch
      end = index_in_epoch #end很简单，就是 index_in_epoch加上batch_size
      return X_train[start:end], y_train[start:end] #在数据x,y


def train(X_train, X_test, y_train, y_test):
    # 定义输入placeholder
    x = tf.placeholder(tf.float32, [None, heli_inference.INPUT_NODE],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, heli_inference.OUTPUT_NODE],
                        name='y-input')
    # 定义正则化器及计算前向过程输出
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = heli_inference.inference(x, regularizer)
    # 定义当前训练轮数及滑动平均模型
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               TRAINING_NUMS / BATCH_SIZE, LEARNING_RATE_DECAY)
    # 定义训练操作，包括模型训练及滑动模型操作
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variables_averages_op)
    # 定义Saver类对象，保存模型，TensorFlow持久化类
    saver = tf.train.Saver()

    # 定义会话，启动训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epochs_completed=0
        index_in_epoch=0
        for i in range(TRAINING_STEPS):
            xs,ys=next_batch(X_train,y_train,epochs_completed,index_in_epoch,BATCH_SIZE)

            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." \
                      % (step, loss_value))
                # save方法的global_step参数可以让每个被保存的模型的文件名末尾加上当前训练轮数
                # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                #            global_step=global_step)

        export_model(tf.train.Saver(), ["layer1/input_node"], "layer2/output_node")

def main(argv=None):
    X_list, y_list=tf_utils.read_data()
    X_train, X_test, y_train, y_test = tf_utils.split_data(X_list, y_list)
    print("split ok")
    train(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    tf.app.run()
