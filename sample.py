from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

import numpy
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to load stored checkpointed models from')
    parser.add_argument('-n', type=int, default=120,
                       help='number of words to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                       help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=5,
                       help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            test_str = model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width)

            w = test_str.split()

            i = 0
            text_file = open("Output.txt", "w")

            L = random.sample([4,4,6,8], 4)
            l = 0
            count = 0

            while(i<len(w)):
                n = numpy.random.choice(numpy.arange(1, 8), p=[0.01, 0.01, 0.04, 0.3, 0.3, 0.24,0.1])

                line = ' '.join(w[i:i+n])
                
                line = line.replace("\"","")
                line = line.replace("[","")
                line = line.replace("]","")
                line = line.replace("(","")
                line = line.replace(")","")
                line = line.replace(".","")

                i = i + n

                line = line.capitalize()

                text_file.write(line)

                count = count + 1
                if (count==L[l] and l<3):
                    text_file.write('.' + '\n')
                    count = 0
                    l = l + 1


                text_file.write('\n')

            text_file.close()

            text_file = open("Output.txt","r")
            data = text_file.read()
            print('\n' + data)
            text_file.close()

if __name__ == '__main__':
    main()
