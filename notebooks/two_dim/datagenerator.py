import random
import numpy as np
from tifffile import imread


def data_generator_2D(data_config):
    GT_image_dr = data_config['GT_image_dr']
    lowSNR_image_dr = data_config['lowSNR_image_dr']
    patch_size = data_config['patch_size']
    n_patches = data_config['n_patches']
    n_channel = data_config['n_channel']
    threshold = data_config['threshold']
    lp = data_config['lp']
    augment = data_config['augment']
    shuffle = data_config['shuffle']
    add_noise = data_config['add_noise']

    gt = imread(GT_image_dr).astype(np.float64)
    low = imread(lowSNR_image_dr).astype(np.float64)
    if len(gt.shape) == 3:
        gt = np.reshape(gt, (gt.shape[0], 1, gt.shape[1], gt.shape[2]))
        low = np.reshape(low, (low.shape[0], 1, low.shape[1], low.shape[2]))
    if len(gt.shape) == 2:
        gt = np.reshape(gt, (1, 1, gt.shape[0], gt.shape[1]))
        low = np.reshape(low, (1, 1, low.shape[0], low.shape[1]))
    print(gt.shape)
    m = gt.shape[0]
    img_size = gt.shape[2]

    gt = gt / (gt.max(axis=(-1, -2))).reshape((gt.shape[0], gt.shape[1], 1, 1))
    low = low / (low.max(axis=(-1, -2))).reshape((low.shape[0], low.shape[1], 1, 1))

    if add_noise:
        for i in range(len(gt)):
            low[i] = np.random.poisson(gt[i] / lp, size=gt[i].shape)

    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    y = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)

    rr = np.floor(np.linspace(0, img_size - patch_size, n_patches)).astype(np.int32)
    cc = np.floor(np.linspace(0, gt.shape[3] - patch_size, n_patches)).astype(np.int32)

    count = 0
    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                x[count, :, :, 0] = low[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                y[count, :, :, 0] = gt[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                count = count + 1

    if augment:
        count = x.shape[0]
        xx = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)
        yy = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)

        xx[0:count, :, :, :] = x
        xx[count:2 * count, :, :, :] = np.flip(x, axis=1)
        xx[2 * count:3 * count, :, :, :] = np.flip(x, axis=2)
        xx[3 * count:4 * count, :, :, :] = np.flip(x, axis=(1, 2))

        yy[0:count, :, :, :] = y
        yy[count:2 * count, :, :, :] = np.flip(y, axis=1)
        yy[2 * count:3 * count, :, :, :] = np.flip(y, axis=2)
        yy[3 * count:4 * count, :, :, :] = np.flip(y, axis=(1, 2))
    else:
        xx = x
        yy = y

    norm_x = np.linalg.norm(yy, axis=(1, 2))
    norm_x = norm_x / norm_x.max()
    ind_norm = np.where(norm_x > threshold)[0]
    print(len(ind_norm))

    xxx = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))
    yyy = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))

    for i in range(len(ind_norm)):
        xxx[i] = xx[ind_norm[i]]
        yyy[i] = yy[ind_norm[i]]

    aa = np.linspace(0, len(xxx) - 1, len(xxx))
    random.shuffle(aa)
    aa = aa.astype(int)

    xxs = np.empty(xxx.shape, dtype=np.float64)
    yys = np.empty(yyy.shape, dtype=np.float64)

    if shuffle:
        for i in range(len(xxx)):
            xxs[i] = xxx[aa[i]]
            yys[i] = yyy[aa[i]]
    else:
        xxs = xxx
        yys = yyy

    xxs[xxs < 0] = 0
    xxs = xxs / (xxs.max(axis=(1, 2))).reshape((xxs.shape[0], 1, 1, 1))
    yys = yys / (yys.max(axis=(1, 2))).reshape((yys.shape[0], 1, 1, 1))

    x_train = xxs
    y_train = yys

    print('Dataset shape:', x_train.shape)
    return x_train, y_train


