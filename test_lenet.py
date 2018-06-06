import tensorflow as tf
from lenet import LeNet
import matplotlib.pyplot as plt

image_size = 28
channels = 1
keep_prob = 0.8
num_classes = 10
batch_size = 128
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def test_image(filename, num_class, weights_path='Default'):
    img_string = tf.gfile.FastGFile(filename, 'rb').read()
    img_decoded = tf.image.decode_jpeg(img_string, channels=channels)
    img_resized = tf.image.resize_images(img_decoded, [image_size, image_size])
    img_reshape = tf.reshape(img_resized, shape=[1, image_size, image_size, channels])

    model = LeNet(img_reshape, keep_prob, num_classes, batch_size, image_size, channels)
    score = tf.nn.softmax(model.fc6)
    max = tf.argmax(score, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./tmp/checkpoint/model_epoch10.ckpt")
        print(sess.run(model.fc6))
        prob = sess.run(max)[0]
        print("The number is %s"%(class_name[prob]))

        #plt.imshow(img_decoded.eval())
        #plt.title("Class: " + class_name[prob])
        #plt.show()

test_image("./test/0009.jpg", num_class=10)