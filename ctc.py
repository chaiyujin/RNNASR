import sys
import time
from utils.process_bar import process_bar
from utils.sample_draw import sample_draw
import tensorflow as tf
from data_loader import load_data, process_target, mean_std, pad_sequences
from data_loader import load_single
import matplotlib.pyplot as plt


def LSTMCell(num_hidden):
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    return cell


# use 13 mfcc
num_features = 39
# 61 phonemes and blank
num_classes = 40

num_hidden = 128
num_layers = 1
num_epochs = 1
batch_size = 32
# initial_learning_rate = 5e-3
initial_learning_rate = 1e-3

# load all data from pkl
all_data = load_data()
# process training data

train_set = all_data['train_set']
test_set = all_data['test_set']

mean, std = mean_std(train_set['sources'])
train_set['sources'] = (train_set['sources'] - mean) / std
test_set['sources'] = (test_set['sources'] - mean) / std

# padding
train_set['sources'], train_set['seq_len'] = pad_sequences(train_set['sources'])
test_set['sources'], test_set['seq_len'] = pad_sequences(test_set['sources'])

num_examples = batch_size
# num_val = 1
# num_examples = len(train_set['sources'])
num_val = len(test_set['sources'])
num_batches_per_epoch = int(num_examples/batch_size)

# data to draw
one_data = load_single()

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

    phoneme_prob = tf.nn.softmax(logits)

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

epochs = []
train_costs = []
valid_costs = []
train_lers = []
valid_lers = []

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    val_cost = float('nan')
    val_ler = float('nan')

    for curr_epoch in range(num_epochs):
        # the No. curr_epoch
        epochs.append(curr_epoch)

        # training phase
        train_cost = train_ler = 0
        start = time.time()
        print("train...")
        for batch in range(num_batches_per_epoch):
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
            batch_train_inputs = train_set['sources'][indexes]
            batch_train_seq_lens = train_set['seq_len'][indexes]
            batch_train_targets = process_target(train_set['targets'][indexes])

            # batch_train_inputs, batch_train_seq_lens = pad_sequences(batch_train_inputs)

            feed = {inputs: batch_train_inputs,
                    targets: batch_train_targets,
                    seq_len: batch_train_seq_lens}

            batch_cost, _ = session.run([cost, optimizer], feed)
            batch_ler = session.run(ler, feed_dict=feed)
            sys.stdout.write('Cost: ' + str(batch_cost) + '\t')
            sys.stdout.write('LER: ' + str(batch_ler) + '\t')

            train_cost += batch_cost * batch_size
            train_ler += batch_ler * batch_size

            process_bar(batch, num_batches_per_epoch)

        train_cost /= num_examples
        train_ler /= num_examples

        # validation phase
        val_targets = process_target(test_set['targets'])
        val_feed = {inputs: test_set['sources'],
                    targets: val_targets,
                    seq_len: test_set['seq_len']}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        # print log
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))

        # draw training curve
        train_costs.append(train_cost)
        train_lers.append(train_ler)
        valid_costs.append(val_cost)
        valid_lers.append(val_ler)

        if curr_epoch % 10 == 1:
            # draw the figure of the training process
            fig = plt.figure(figsize=(12, 12))
            cost_plt = fig.add_subplot(211)
            ler_plt = fig.add_subplot(212)
            cost_plt.title.set_text('Cost')
            ler_plt.title.set_text('LER')
            cost_plt.plot(epochs, train_costs, 'g', epochs, valid_costs, 'r')
            ler_plt.plot(epochs, train_lers, 'g', epochs, valid_lers, 'r')
            plt.savefig('error.png')
            plt.clf()

        # draw alignment
        draw_feed = {inputs: one_data['sources'],
                     seq_len: one_data['seq_len']}
        prob = session.run([phoneme_prob], feed_dict=draw_feed)
        sample_draw(prob[0], one_data['prefix'])

# plt.show()
