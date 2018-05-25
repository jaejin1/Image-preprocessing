from operator import Operator
import densenet

class model(Operator):
    def __init__(self, args, sess):
        Operator.__init__(self, args, sess)

    def model