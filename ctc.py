import time, sys
import utils
import numpy as np
import tensorflow as tf
from dataloader import load_data, process_data, get_phonemes_list_and_map, process_target, pad_sequences


def LSTMCell(num_hidden):
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    return cell


# use 13 mfcc
num_features = 13
# 61 phonemes and blank
num_classes = 62

num_hidden = 50
num_layers = 1
num_epochs = 200
batch_size = 64
initial_learning_rate = 1e-2

# load all data from pkl
all_data = load_data()
# process training data
train_inputs, train_seq_lens, train_targets = \
    process_data(all_data['train_set'])
val_inputs, val_seq_lens, val_targets = \
    process_data(all_data['test_set'])

train_set = np.asarray(all_data['train_set'])
test_set = np.asarray(all_data['test_set'])

num_examples = len(train_set)
num_val = len(test_set)
num_batches_per_epoch = int(num_examples/batch_size)

# build rnn
graph = tf.Graph()
with graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, None, num_features])
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])

    # cell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell(
        [LSTMCell(num_hidden) for _ in range(num_layers)],
        state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    val_cost = float('nan')
    val_ler = float('nan')

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()
        print("train...")
        for batch in range(num_batches_per_epoch):
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
            train_data = train_set[indexes]

            batch_train_inputs, batch_train_seq_lens, batch_train_targets = \
                process_data(train_data)

            batch_train_inputs, batch_train_seq_lens = pad_sequences(batch_train_inputs)

            feed = {inputs: batch_train_inputs,
                    targets: batch_train_targets,
                    seq_len: batch_train_seq_lens}

            batch_cost, _ = session.run([cost, optimizer], feed)
            sys.stdout.write('Cost: ' + str(batch_cost) + '\t')

            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

            utils.process_bar(batch, num_batches_per_epoch)

        train_cost /= num_examples
        train_ler /= num_examples

        if curr_epoch % 20 == 0:
            val_cost_all = 0
            val_ler_all = 0
            print("\nvalid...")
            for i in range(num_val):
                utils.process_bar(i, num_val)
                val_set = test_set[[i]]

                val_inputs, val_seq_lens, val_targets = \
                    process_data(val_set)

                val_feed = {inputs: val_inputs,
                            targets: val_targets,
                            seq_len: val_seq_lens}

                val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

                val_cost_all += val_cost
                val_ler_all += val_ler

            val_cost = val_cost_all / num_val
            val_ler = val_ler_all / num_val

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))

