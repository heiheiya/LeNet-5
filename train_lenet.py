from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from lenet import LeNet
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 1e-3
dropout_rate = 0.8
batch_size = 128
image_size = 28
channels = 1
num_classes = 10
display_step = 200
num_epoches = 10
num_steps = 1000
filewriter_path = "./tmp/tensorboard"
checkpoint_path = "./tmp/checkpoint"

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

model = LeNet(x, keep_prob, num_classes, batch_size, image_size, channels)
prediction = model.fc6

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y, name='cross_entropy_per_example'))
loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
format_str = ('epoch %d, step %d: loss=%.2f, training accuracy=%.3f')

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(filewriter_path, sess.graph)
saver = tf.train.Saver()

tf.global_variables_initializer().run()
tf.train.start_queue_runners()

print("{} Start training...".format(datetime.now()))
print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

for epoch in range(num_epoches):
    for step in range(num_steps):
        batch = mnist.train.next_batch(batch_size)
        if step%display_step == 0:
            losses = sess.run(loss, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            writer.add_summary(s, epoch * num_steps + step)
            print(format_str%(epoch+1, step+1, losses, train_acc))
        sess.run(train_op, feed_dict={x: batch[0], y: batch[1], keep_prob: dropout_rate})
    test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
    print("epoch %d: test accuracy=%.3f"%(epoch+1, test_acc))

    print("{} Saving checkpoint of model...".format(datetime.now()))

    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
    save_path = saver.save(sess, checkpoint_name)

writer.close()