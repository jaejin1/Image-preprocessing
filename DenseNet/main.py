import argparse
import distutils.util
import os
import tensorflow as tf
import model as densenet

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--flag", type=distutils.util.strtobool, default='true')
    parser.add_argument("-g", "--gpu_number", type=str, default="0")
    parser.add_argument("-p", "--project", type=str, default="R&D")

    # Train Data
    parser.add_argument("-d", "--data_dir", type=str, default="./Data")
    parser.add_argument("-trs", "--data_size", type=int, default=64)

    # Train Parameter
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-lr", "--learning_rate", type=float, default= 1e-4)
    parser.add_argument("-m", "--momentum", tpye=float, ddefault=0.5)
    parser.add_argument("-m2", "--momentum2", tpye=float, default=0.999)

    args = parser.parse_args()

    gpu_number = args.gpu_number
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    with tf.device('/gpu:{0}'.format(gpu_number)):
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_option=gpu_option)

        with tf.Session(config=config) as sess:

            model = densenet.model(args, sess)

            # Trina / Test

            if args.flag:
                model.train(args.flag)
            else:
                model.test(args.flag)

if __name__ == '__main__':
    main()
