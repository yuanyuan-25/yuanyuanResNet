import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from resnet import  resnet10

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)


# 加载数据
def preprocess(x, y):
    # x转化为float32的数据类型，并通过/255缩放到0-1，再-1缩放到-1-0
    x = tf.cast(x, dtype=tf.float32) / 255. - 1 #灰度归一化
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()  # 两个多元组返回值
# y值降维到1维
y = tf.squeeze(y, axis=1)  # [n, 1] => [n]
y_test = tf.squeeze(y_test, axis=1)  # [n, 1] => [n]
#特征值和标签一一匹配
train_db = tf.data.Dataset.from_tensor_slices((x, y))
#每1000个作为一组进行打乱，map用参数里的函数对数据进行预处理。batchsize为256，每256个数据进行一次参数更新。
train_db = train_db.shuffle(1000).map(preprocess).batch(256)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(256)


def main():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet10()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(1e-4)
#对每个迭代样本进行梯度计算
    for epoch in range(50): # 对一整个训练集进行50次epoch
        for step, (x, y) in enumerate(train_db):#每一个epoch里有datasize/256个batch，对每一个batch进行梯度计算
            with tf.GradientTape() as tape:
                #当前batch的预测值
                logits = model(x, training=True)

                y_onehot = tf.one_hot(y, depth=100)
                #交叉熵损失函数，用于多分类问题
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
# 梯度带loss对变量进行求导，trainable_variables=所有的变量list
            grads = tape.gradient(loss, model.trainable_variables)
            # 把求出的梯度下降到变量上，用于更新变量
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
                #输出当前是第几个epoch，的第几个beach，的损失函数
        total_num = 0
        total_correct = 0
        #这里是在每个epoch结束进行测试
        for x, y in test_db:
            #存放模型的输出结果
            logits = model(x, training=False)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
