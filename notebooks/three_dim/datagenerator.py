import random
import glob
import numpy as np
from tifffile import imread


def stack_generator_3D(GT_dr, low_dr, fr_start, fr_end):
    path_gt = GT_dr + '/*.tif'
    path_low = low_dr + '/*.tif'
    image_gt = imread(sorted(glob.glob(path_gt))).astype(np.float32)
    image_low = imread(sorted(glob.glob(path_low))).astype(np.float32)

    if len(image_gt.shape) == 3:
        image_gt = np.reshape(image_gt, (image_gt.shape[0], 1, 1, image_gt.shape[1], image_gt.shape[2]))
        image_low = np.reshape(image_low,
                               (image_low.shape[0], 1, 1, image_low.shape[1], image_low.shape[2]))

    if len(image_gt.shape) == 4:
        image_gt = np.reshape(image_gt, (image_gt.shape[0], image_gt.shape[1], 1, image_gt.shape[2], image_gt.shape[3]))
        image_low = np.reshape(image_low,
                               (image_low.shape[0], image_low.shape[1], 1, image_low.shape[2], image_low.shape[3]))

    print(image_gt.shape)
    for i in range(len(image_gt)):
        for j in range(image_gt.shape[2]):
            if image_gt[i, :, j, :, :].max() > 0:
                image_gt[i, :, j, :, :] = image_gt[i, :, j, :, :] / image_gt[i, :, j, :, :].max()
            if image_low[i, :, j, :, :].max() > 0:
                image_low[i, :, j, :, :] = image_low[i, :, j, :, :] / image_low[i, :, j, :, :].max()

    crop_gt = image_gt[:, fr_start:fr_end, :, :, :]
    crop_low = image_low[:, fr_start:fr_end, :, :, :]
    crop_gt = np.moveaxis(crop_gt, 1, -1)
    crop_low = np.moveaxis(crop_low, 1, -1)
    crop_gt = np.moveaxis(crop_gt, 1, -1)
    crop_low = np.moveaxis(crop_low, 1, -1)
    print(crop_low.shape)
    return crop_gt, crop_low


def data_generator_3D(data_config):
    GT_dr = data_config['GT_image_dr']
    low_dr = data_config['lowSNR_image_dr']
    patch_size = data_config['patch_size']
    n_patches = data_config['n_patches']
    n_channel = data_config['n_channel']
    threshold = data_config['threshold']
    fr_start = data_config['fr_start']
    fr_end = data_config['fr_end']
    lp = data_config['lp']
    augment = data_config['augment']
    shuffle = data_config['shuffle']
    add_noise = data_config['add_noise']

    gt, low = stack_generator_3D(GT_dr, low_dr, fr_start, fr_end)

    m = gt.shape[0]
    img_size = gt.shape[2]
    z_depth = fr_end - fr_start

    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, z_depth, 1), dtype=np.float32)
    y = np.empty((m * n_patches * n_patches, patch_size, patch_size, z_depth, 1), dtype=np.float32)

    if n_patches == 1:
        rr = [0]
        cc = [0]
    else:
        rr = np.floor(np.linspace(0, img_size - patch_size, n_patches))
        cc = rr
        rr = rr.astype(np.int32)
        cc = cc.astype(np.int32)

    count = 0

    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                x[count, :, :, :, 0] = low[l, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size, :, n_channel]
                y[count, :, :, :, 0] = gt[l, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size, :, n_channel]
                count = count + 1

    # Generating a noisy dataset by adding Poisson noise
    if add_noise:
        for i in range(len(x)):
            x[i] = np.random.poisson((y[i]) / lp, size=y[i].shape)
            # for k in range(x.shape[3]):
            #     pixel_x = np.random.randint(0, patch_size, [2, 10])
            #     pixel_y = np.random.randint(0, patch_size, [2, 10])
            #     x[i, pixel_x[1, :], pixel_y[1, :], k, :] = 0
            #     x[i, pixel_x[0, :], pixel_y[0, :], k, :] = 1

    if augment:
        count = x.shape[0]
        xx = np.zeros((4 * count, patch_size, patch_size, z_depth, 1), dtype=np.float32)
        yy = np.zeros((4 * count, patch_size, patch_size, z_depth, 1), dtype=np.float32)

        xx[0:count, :, :, :, :] = x
        xx[count:2 * count, :, :, :, :] = np.flip(x, axis=1)
        xx[2 * count:3 * count, :, :, :, :] = np.flip(x, axis=2)
        xx[3 * count:4 * count, :, :, :, :] = np.flip(x, axis=(1, 2))

        yy[0:count, :, :, :, :] = y
        yy[count:2 * count, :, :, :, :] = np.flip(y, axis=1)
        yy[2 * count:3 * count, :, :, :, :] = np.flip(y, axis=2)
        yy[3 * count:4 * count, :, :, :, :] = np.flip(y, axis=(1, 2))
    else:
        xx = x
        yy = y

    xx = xx / xx.max()
    yy = yy / yy.max()

    norm_x = np.linalg.norm(np.max(xx, axis=3), axis=(1, 2))
    ind_norm = np.where(norm_x >= threshold)[0]
    print(len(ind_norm))

    xxx = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3], xx.shape[4]))
    yyy = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3], xx.shape[4]))

    for i in range(len(ind_norm)):
        xxx[i] = xx[ind_norm[i]]
        yyy[i] = yy[ind_norm[i]]

    aa = np.linspace(0, len(xxx) - 1, len(xxx))
    random.shuffle(aa)
    aa = aa.astype(int)

    xxs = np.empty(xxx.shape, dtype=np.float32)
    yys = np.empty(yyy.shape, dtype=np.float32)

    if shuffle:
        for i in range(len(xxx)):
            xxs[i] = xxx[aa[i]]
            yys[i] = yyy[aa[i]]
    else:
        xxs = xxx
        yys = yyy

    x_train = xxs
    y_train = yys

    print('Dataset shape:', x_train.shape)
    return x_train, y_train
