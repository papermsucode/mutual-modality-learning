# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/ssd/ssd_srv80/video/datasets/'

def return_somethingv2(modality):
    filename_categories = 'Something-Something-v2/lists/category.txt'
    filename_imglist_train = []
    filename_imglist_val = []
    root_data = []
    prefix = []

    for moda in modality:
        if moda == 'RGB' or moda == 'RGBDiff':
            root_data.append(ROOT_DATASET + 'Something-Something-v2/images')
            filename_imglist_train.append('Something-Something-v2/lists/train_videofolder.txt')
            filename_imglist_val.append('Something-Something-v2/lists/val_videofolder.txt')
            prefix.append('img_{:06d}.jpg')
        elif moda == 'Flow':
            root_data.append(ROOT_DATASET + 'Something-Something-v2/flow_images')
            filename_imglist_train.append('Something-Something-v2/lists/train_videofolder_flow.txt')
            filename_imglist_val.append('Something-Something-v2/lists/val_videofolder_flow.txt')
            prefix.append('flow_{}_{:05d}.jpg')
        else:
            raise NotImplementedError('no such modality:'+modality)

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    modality = modality.split(',')
    #dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
    #               'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
    #               'kinetics': return_kinetics }
    dict_single = {'somethingv2': return_somethingv2}

    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = [os.path.join(ROOT_DATASET, x) for x in file_imglist_train]
    file_imglist_val = [os.path.join(ROOT_DATASET, x) for x in file_imglist_val]

    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
