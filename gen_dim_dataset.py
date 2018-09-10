# generate dataset configuration file for block dimension(L,W,H) estimation
# from basic dataset
# created by QIU Xuchong
# 2018/08

import argparse  # module for user-friendly command-line interfaces
import pandas as pd  # easy csv parsing
import numpy as np
import matplotlib.pyplot as plt  # for visualization
import time
import warnings


# command-line interface arguments
parser = argparse.ArgumentParser(description='Script for generating block dimension estimation dataset')
parser.add_argument('--csv_path', default='/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/'
                                          'scenario_PV3.1/CSV_files/20171214-104147_data.txt',
                    type=str, help='directory containing dataset csv files')
parser.add_argument('--numClasses', default=[10, 10, 10], type=int,
                    help='number of class for each classification task')
parser.add_argument('--bin_size', default=7.5, type=int,
                    help='bin size of classification in regression task ')
parser.add_argument('--train_ratio', default=0.8, type=int,
                    help='ratio of training set on whole dataset')


def main():
    global args
    args = parser.parse_args()
    samples = pd.read_csv(args.csv_path)

    # fetch sample path and dimension labels
    path = samples['{path}']  # or samples.path
    block_length = samples['{BlockSizeX}']
    block_width = samples['{BlockSizeY}']
    block_height = samples['{BlockSizeZ}']

    # histogram of dimension classes in original dataset
    block_length.hist(bins=50)
    plt.xlabel('block size x(cm)')
    plt.show()
    block_width.hist(bins=50)
    plt.xlabel('block size y(cm)')
    plt.show()
    block_height.hist(bins=50)
    plt.xlabel('block size Z(cm)')
    plt.show()

    # assign new labels for new dataset configuration file
    size_x_label = block_length*10
    size_y_label = block_width*10
    size_z_label = block_height*10
    imgs_path = path.str.replace('\\', '/')

    dim_min = 25
    dim_max = 100

    for class_idx in range(0, args.numClasses[0]):
        interval_min = dim_min + args.bin_size*class_idx
        interval_max = dim_min + args.bin_size*(class_idx+1)

        size_x_label[(size_x_label >= interval_min) & (size_x_label < interval_max)] = class_idx
        size_y_label[(size_y_label >= interval_min) & (size_y_label < interval_max)] = class_idx
        size_z_label[(size_z_label >= interval_min) & (size_z_label < interval_max)] = class_idx

    size_x_label[size_x_label == dim_max] = (args.numClasses[0] - 1)
    size_y_label[size_y_label == dim_max] = (args.numClasses[0] - 1)
    size_z_label[size_z_label == dim_max] = (args.numClasses[0] - 1)

    # histogram of dimension classes in dimension estimation dataset
    size_x_label.hist()
    plt.xlabel('block size x on dataset')
    plt.show()
    size_y_label.hist()
    plt.xlabel('block size y on dataset')
    plt.show()
    size_z_label.hist()
    plt.xlabel('block size z on dataset')
    plt.show()

    # save in dataset csv file
    df_dim = pd.DataFrame({'path': imgs_path,
                           'block_dim_x': size_x_label,
                           'block_dim_y': size_y_label,
                           'block_dim_z': size_z_label})

    # calculate class weights and save
    size_x_weight = np.zeros([args.numClasses[0]])
    size_y_weight = np.zeros([args.numClasses[1]])
    size_z_weight = np.zeros([args.numClasses[2]])

    for class_idx in range(0, args.numClasses[0]):
        size_x_weight[class_idx] = len(size_x_label) / sum(size_x_label == class_idx)
        size_y_weight[class_idx] = len(size_y_label) / sum(size_y_label == class_idx)
        size_z_weight[class_idx] = len(size_z_label) / sum(size_z_label == class_idx)

    df_weights = pd.DataFrame({'dim_x_weight': size_x_weight,
                               'dim_y_weight': size_y_weight,
                               'dim_z_weight': size_z_weight})

    # split training/validation set
    msk = np.random.rand(len(df_dim)) < args.train_ratio
    df_dim_train = df_dim[msk]
    df_dim_val = df_dim[~msk]

    # histogram for train/val set
    df_dim_train['block_dim_x'].hist(bins=args.numClasses[0])
    plt.xlabel('block size x for train')
    plt.show()
    df_dim_train['block_dim_y'].hist(bins=args.numClasses[0])
    plt.xlabel('block size y for train')
    plt.show()
    df_dim_train['block_dim_z'].hist(bins=args.numClasses[0])
    plt.xlabel('block size for train')
    plt.show()

    df_dim_val['block_dim_x'].hist(bins=args.numClasses[0])
    plt.xlabel('block size x for val')
    plt.show()
    df_dim_val['block_dim_y'].hist(bins=args.numClasses[0])
    plt.xlabel('block size y for val')
    plt.show()
    df_dim_val['block_dim_z'].hist(bins=args.numClasses[0])
    plt.xlabel('block size z for val')
    plt.show()

    date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())  # current date
    df_weights.to_csv(date_time + '_weights.txt', index=False)
    df_dim_train.to_csv(date_time+'_train.txt', index=False)
    df_dim_val.to_csv(date_time + '_val.txt', index=False)


if __name__ == '__main__':
    main()
