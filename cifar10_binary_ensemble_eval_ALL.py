from __future__ import print_function
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, LBFGS, DeepFool, BasicIterativeMethod, \
    CarliniWagnerL2, ProjectedGradientDescent, SaliencyMapMethod
from tensorflow.keras.datasets import cifar10
from cleverhans.utils_tf import model_eval
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from utils_ensemble import *
from models import *
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)


FLAGS.indv_CE_lamda = 10.0
FLAGS.log_det_lamda = 0.0
FLAGS.dataset = 'cifar10'
FLAGS.augmentation = True
subtract_pixel_mean = True
FLAGS.num_models = 30
Dense_code = True


att_para = (['FGSM', [0.0, 0.04]],
            ['BIM', [0.04]],
            ['PGD', [0.04]],
            ['JSMA', [0.2]],
            ['CWL2', [1]])

#att_para = (['FGSM', [0.0, 0.04]],
#            ['BIM', [0.04]],
#            ['PGD', [0.04]])

#att_para = (['PGD', [0.04]],
#            ['CWL2', [1]])

# net_name, num of binary nets, best epoch, indv_lamda, div_lamda
# in case _rc*: div_lamda for feature_div; otherwise div_lamda for weight_div
t_cnn2 = (['_cnn3', 20, 115, 5.0, 2.0],
          ['_resnet14_nBN', 20, 165, 5.0, 2.0],
          ['_resnet14', 20, 135, 5.0, 2.0],
          ['_resnet14_sub40', 4, 148, 5.0, 2.0],
          ['_resnet14_sub40', 10, 97, 5.0, 2.0],
          ['_resnet14_sub40', 20, 89, 5.0, 2.0],
          ['_resnet14_sub40', 30, 150, 5.0, 2.0],
          ['_resnet14_sub40', 40, 165, 5.0, 2.0],
          ['_resnet14_nBN_v2', 20, 119, 5.0, 2.0],
          ['_resnet14_nBN_new', 20, 126, 5.0, 2.0],
          ['_resnet14_new', 20, 128, 5.0, 2.0],
          ['_cnn3_new', 16, 174, 5.0, 2.0],
          ['_cnn3_new', 16, 68, 5.0, 0.5],
          ['_cnn3_featDiff_new', 16, 116, 5.0, 0.5],
          ['_resnet8_new', 16, 141, 5.0, 2.0],
          ['_resnet8_new', 16, 23, 5.0, 0.0],
          ['_resnet8_featDiff_new', 16, 138, 5.0, 0.5],
          ['_resnet8_featDiff_new', 16, 176, 5.0, 0.1],
          ['_resnet8_Ent1Feat_new', 16, 139, 5.0, 0.1],
          ['_resnet14_Ent1Feat_new', 20, 180, 5.0, 0.1],
          ['_resnet14_Ent2Feat_new', 20, 175, 5.0, 0.1])

t_cnn3 = (['_resnet16_new_sub40', 4, 126, 5.0, 2.0],
          ['_resnet16_new_sub40', 10, 156, 5.0, 2.0],
          ['_resnet16_new_sub40', 20, 71, 5.0, 2.0],
          ['_resnet16_new_sub40', 30, 122, 5.0, 2.0],
          ['_resnet16_new_sub40', 40, 57, 5.0, 2.0])


x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, 10))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Input image dimensions.
input_shape = x_train.shape[1:]
# Normalize data.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# If subtract pixel mean is enabled
clip_min = 0.0
clip_max = 1.0
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    clip_min -= x_train_mean
    clip_max -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

if __name__ == "__main__":
    sess = tf.Session()
    tf.keras.backend.set_session(sess)

    t = t_cnn3
    test_opts = [0, 1, 2, 3, 4]
    for att_idx in range(len(att_para)):
        att_name = att_para[att_idx][0]
        gammas = att_para[att_idx][1]
        for i in range(len(test_opts)):
            model_idx = test_opts[i]

            net_name = t[model_idx][0]
            FLAGS.num_models = t[model_idx][1]
            nEpoch = t[model_idx][2]
            FLAGS.indv_CE_lamda = t[model_idx][3]
            FLAGS.log_det_lamda = t[model_idx][4]

            dir_pwd = '/scratch/users/ntu/wangcx/DASN/songy/codematrix_v4'
            dir_name = FLAGS.dataset + '_Ensemble_saved_models' + str(FLAGS.num_models) + '_indvCElamda' + str(
                FLAGS.indv_CE_lamda) + '_logdetlamda' + str(FLAGS.log_det_lamda) + '_submean_' + str(subtract_pixel_mean) \
                       + '_dense_' + str(Dense_code) + '_augment_' + str(FLAGS.augmentation) + net_name
            save_dir = os.path.join(dir_pwd, dir_name)
            model_name = 'model.%s.h5' % str(nEpoch).zfill(3)
            filepath = os.path.join(save_dir, model_name)

            for j in range(len(gammas)):
                model_input = Input(shape=(32, 32, 3))
                pred_input = Input(shape=(64,))
                model_dic = {}
                pred_dic = {}
                model_out = []
                shared_dense = shared()
                for i in range(FLAGS.num_models):
                    # model_dic[str(i)] = cnn_model3(model_input, shared_dense=shared_dense)
                    model_dic[str(i)] = resnet_v1(input=model_input, depth=14, num_classes=2, dataset='cifar10', shared_dense=shared_dense)
                    pred_dic[str(i)] = predictor(pred_input, shared_dense=shared_dense)
                    model_out.append(model_dic[str(i)].output)
                model_output = tf.keras.layers.concatenate(model_out)
                model_output = decoder(model_output,
                                       opt='dense',
                                       drop_prob=0.2)
                loaded_model = Model(model_input, model_output)
                loaded_model.load_weights(filepath)

                wrap = KerasModelWrapper(loaded_model)
                if att_name == 'FGSM':
                    attack = FastGradientMethod(wrap, sess=sess)
                    attacker_params = {'eps': gammas[j],
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'BIM':
                    attack = BasicIterativeMethod(wrap, sess=sess)
                    attacker_params = {'eps': gammas[j],
                                       'eps_iter': 0.02,
                                       'nb_iter': 10,
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'PGD':
                    attack = ProjectedGradientDescent(wrap, sess=sess)
                    attacker_params = {'eps': gammas[j],
                                       'eps_iter': 0.02,
                                       'nb_iter': 10,
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'JSMA':
                    attack = SaliencyMapMethod(wrap, sess=sess)
                    attacker_params = {'gamma': gammas[j],
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'CWL2':
                    attack = CarliniWagnerL2(wrap, sess=sess)
                    attacker_params = {'batch_size': 100,
                                       'confidence': gammas[j],
                                       'learning_rate': 5e-3,
                                       'binary_search_steps': 5,
                                       'clip_min': clip_min,
                                       'clip_max': clip_max
                                       }
                x_adv = attack.generate(x, **attacker_params)
                x_adv = tf.stop_gradient(x_adv)

                eval_par = {'batch_size': 100}
                preds = loaded_model(x_adv)
                acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
                print('att_name: %s, att_para= %.2f, net: %s, adv_acc: %.3f, using %d models, detLamd=%.2f, indvLamd=%.2f' %
                      (att_name, gammas[j], net_name, acc, FLAGS.num_models, FLAGS.log_det_lamda, FLAGS.indv_CE_lamda))






