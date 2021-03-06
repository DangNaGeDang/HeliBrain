import tensorflow as tf

INPUT_NODE = 254
OUTPUT_NODE = 11
LAYER1_NODE = 60


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        # 将权重参数的正则化项加入至损失集合
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases,name="input_node")

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.add(tf.matmul(layer1, weights),biases,name="output_node")

    return layer2
