import glob
import time
import numpy as np
import tensorflow as tf
import os.path

class op_base:
    def __init__(self, args, sess):
        self.sess = sess

        # Train
        self.flag = args.flag
        self.gpu_number = args.gpu_number
        self.project = args.project

        # Train Data
        self.data_dir = args.data_dir
        self.data_size = args.data_size

        # Train parameter
        self.batch_size = args.batch_size
        self.learning_rate = args.lerarning_rate
        self.mm = args.momentum
        self.mm2 = args.momentum2

        # Result Dir
        self.project_dir = 'assets/{0}_{1}/'.format(self.project, self.data_size)
        self.ckpt_dir = os.path.join(self.project_dir, 'models')
        self.model_name = "{0}.model".format(self.project)
        self.ckpt_model_name = os.path.join(self.ckpt_dir, self.model_name)

        # etc
        if not os.path.exists('assets'):
            os.makedirs('assets')
        self.make_project_dir(self.project_dir)

    def load(selfself, sess, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))


    def make_project_dir(project_dir):
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
            os.makedirs(os.path.join(project_dir, 'models'))
            os.makedirs(os.path.join(project_dir, 'result'))
            os.makedirs(os.path.join(project_dir, 'result_test'))



