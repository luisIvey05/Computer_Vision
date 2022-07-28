import numpy as np

def preprocess_image(img):
    '''
    Using the same normalization parameters used during training, the
    passed in image is normalized using those very same parameters.

    :param img:
    img - a rgb image
    :return:
    img - a normalized image
    '''
    IMG_SCALE = 1./255
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


def depth_to_rgb(depth):
    '''
    Converts input depth map to a rgb image.
    :param depth:
    :return: colormapped_im
    '''
    normalizer = co.Normalize(vmin=0, vmax=180)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='cividis')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)

    return colormapped_im