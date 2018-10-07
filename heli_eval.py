import time
import tensorflow as tf
import heli_inference
import heli_train
import tf_utils
EVAL_INTERVAL_SECS = 10


def evaluate(X_train, X_test, y_train, y_test):
    with tf.Graph().as_default() as g:
        # 定义输入placeholder
        x = tf.placeholder(tf.float32, [None, heli_inference.INPUT_NODE],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, heli_inference.OUTPUT_NODE],
                            name='y-input')
        # 定义feed字典
        validate_feed = {x: X_test, y_: y_test}
        # 测试时不加参数正则化损失
        y = heli_inference.inference(x, None)
        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 加载滑动平均模型下的参数值
        variable_averages = tf.train.ExponentialMovingAverage(
            heli_train.MOVING_AVERAGE_DECAY)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        # 每隔EVAL_INTERVAL_SECS秒启动一次会话
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(heli_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 取checkpoint文件中的当前迭代轮数global_step
                    global_step = ckpt.model_checkpoint_path \
                        .split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" \
                          % (global_step, accuracy_score))

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    X_list, y_list=tf_utils.read_data()
    X_train, X_test, y_train, y_test = tf_utils.split_data(X_list, y_list)
    evaluate(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    tf.app.run()
