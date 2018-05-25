import glob
import time
import datetime
import cv2

from operator.operator_base import op_base

class Operator(op_base):
    def __init__(self, args, sess):
        op_base.__init__(self, args, sess)
        self.build_model()

    def build_model(self):
        print('temp')
        # 각종 input placeholder
        # 모델 연산들
        # loss 계산
        # variables
        # Optimizer
        # initializer
        # tf saver
        # summary

    def train(self, train_flag):
        print('temp')
        # train

    def test(self, train_flag=True):
        print('temp')
        # test