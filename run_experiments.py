from __future__ import print_function
import matplotlib
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf2
import random as rn
### We modified Pahikkala et al. (2014) source code for cross-val process ###
# For our implementation, the DeepDTA source code (https://github.com/hkmztrk/DeepDTA) was adopted with some required modifications and customization.#

import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
print(keras.__version__)

from datahelper import *
from arguments import argparser, logging
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input, Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import json
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from emetrics import get_aupr, get_cindex, get_rm2

TABSY = "\t"
figdir = "figures/"


def inception_block(input_tensor, filters):
    conv1 = Conv1D(filters[0], kernel_size=4, activation='relu', padding='same')(input_tensor)

    conv2 = Conv1D(filters[1], kernel_size=8, activation='relu', padding='same')(input_tensor)

    conv3 = Conv1D(filters[2], kernel_size=16, activation='relu', padding='same')(input_tensor)

    inception_output = keras.layers.concatenate([conv1, conv2, conv3], axis=-1)

    return inception_output


# ------------

def inceptionDTA(FLAGS, NUM_FILTERS):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len, 20,), dtype='float32')

    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    encode_smiles = inception_block(encode_smiles, filters=[NUM_FILTERS, NUM_FILTERS * 2, NUM_FILTERS * 3])
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = inception_block(XTinput, filters=[NUM_FILTERS, NUM_FILTERS * 2, NUM_FILTERS * 3])
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1)

    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    predictions = Dense(1, kernel_initializer='normal')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])
    print(interactionModel.summary())
    return interactionModel


def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, measure, runmethod, FLAGS, dataset):
    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS)

    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    # logger.info('Start training')
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT, Y,
                                                                                                          label_row_inds,
                                                                                                          label_col_inds,
                                                                                                          measure,
                                                                                                          runmethod,
                                                                                                          FLAGS,
                                                                                                          train_sets,
                                                                                                          val_sets)

    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, Y, label_row_inds,
                                                                                         label_col_inds,
                                                                                         measure, runmethod, FLAGS,
                                                                                         train_sets, test_sets)

    testperf = all_predictions[bestparamind]  ##pointer pos

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    print("length of test is :")
    print(len(test_sets))
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd


def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets,
                     val_sets):
    from dataset import DataGenerator
    paramset1 = FLAGS.num_windows  # [32]#[32,  512] #[32, 128]  # filter numbers

    epoch = FLAGS.num_epoch
    # if flag==0:                                #100
    batchsz = FLAGS.batch_size  # 256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)
    fpath = FLAGS.dataset_path
    CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                   "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                   "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                   "U": 19, "T": 20, "W": 21,
                   "V": 22, "Y": 23, "X": 24,
                   "Z": 25}
    index2char = {index: char for char, index in CHARPROTSET.items()}
    with open(fpath + "deep_prime_to_sec_protovec_encoding.json") as f:
        pro2vec = json.load(f)
    max_prot_len = FLAGS.max_seq_len
    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = XD[trrows]
        XT_train = XT[trcols]

        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]

        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        pointer = 0
        train_generator = DataGenerator(train_drugs, train_prots, train_Y, pro2vec, batchsz, max_prot_len, index2char,
                                        shuffle=False)
        validation_generator = DataGenerator(val_drugs, val_prots, val_Y, pro2vec, batchsz, max_prot_len, index2char,
                                             shuffle=False)
        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]

            gridmodel = runmethod(FLAGS, param1value)
            es = EarlyStopping(monitor='val_cindex_score', mode='max', verbose=1, patience=15,
                               restore_best_weights=True, )
            mc = ModelCheckpoint('best_modele_kiba.h5', monitor='val_cindex_score', mode='max', verbose=1,
                                 save_best_only=True)
            gridres = gridmodel.fit_generator(generator=train_generator, validation_data=validation_generator,
                                              epochs=epoch, callbacks=[es, mc])
            predicted_labels = gridmodel.predict_generator(validation_generator, verbose=0)
            loss, rperf2 = gridmodel.evaluate_generator(validation_generator, verbose=0)
            rperf = prfmeasure(val_Y, predicted_labels)

            rperf = rperf[0]

            logging("P1 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" %
                    (param1ind, foldind, rperf, rperf2, loss), FLAGS)

            all_predictions[pointer][foldind] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
            all_losses[pointer][foldind] = loss

            pointer += 1

    bestperf = -float('Inf')
    bestpointer = None
    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):

        avgperf = 0.
        for foldind in range(len(val_sets)):
            foldperf = all_predictions[pointer][foldind]
            avgperf += foldperf
        avgperf /= len(val_sets)
        # print(epoch, batchsz, avgperf)
        if avgperf > bestperf:
            bestperf = avgperf
            bestpointer = pointer
            best_param_list = [param1ind]

        pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf2.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g / f)  # select


def plotLoss(history, batchind, epochind, param3ind, foldind):
    figname = "b" + str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_" + str(foldind) + "_" + str(
        time.time())
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/" + figname + ".png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()
    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/" + figname + "_acc.png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6):  # 5-fold cross validation + test

    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation

    dataset = DataSet(fpath=FLAGS.dataset_path,  ### BUNU ARGS DA GUNCELLE
                      setting_no=FLAGS.problem_type,  ##BUNU ARGS A EKLE
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)  # basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                    perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    perfmeasure = get_cindex
    deepmethod = inceptionDTA

    experiment(FLAGS, perfmeasure, deepmethod)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression(FLAGS)
