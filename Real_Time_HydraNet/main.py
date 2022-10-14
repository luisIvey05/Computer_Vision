import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import os
import shutil
import matplotlib.cm as cm
import matplotlib.colors as co
import numpy as np
import glob
import cv2
import argparse
import sys
from utils.hydra import HydraNet
from utils.helper import preprocess_image
from PIL import Image


# wget https://hydranets-data.s3.eu-west-3.amazonaws.com/hydranets-data.zip && unzip -q hydranets-data.zip && mv hydranets-data/* . && rm hydranets-data.zip && rm -rf hydranets-data

def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def pipeline(img, hydranet, NUM_CLASSES, CMAP):
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(preprocess_image(img).transpose(2, 0, 1)[None]),
                           requires_grad=False).float()  # Put the image in PyTorch Variable
        if torch.cuda.is_available():
            img_var = img_var.cuda()  # Send to GPU
        segm, depth = hydranet(img_var)  # Call the HydraNet

        segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0), img.shape[:2][::-1],
                          interpolation=cv2.INTER_CUBIC)  # Post-Process / Resize

        depth = cv2.resize(depth[0, 0].cpu().data.numpy(), img.shape[:2][::-1],
                           interpolation=cv2.INTER_CUBIC)  # Post-Process / Resize

        segm = CMAP[segm.argmax(axis=2).astype(np.uint8)]  # Use the CMAP

        depth = np.abs(depth)  # Take the abs value

    return depth, segm


def main(source, fpath):
    hydranet = HydraNet()
    hydranet.define_mobilenet()
    hydranet.define_lightweight_refinenet()

    if torch.cuda.is_available():
        _ = hydranet.cuda()
    hydranet.eval()

    ckpt = torch.load('ExpKITTI_joint.ckpt')
    hydranet.load_state_dict(ckpt['state_dict'])

    CMAP = np.load('cmap_kitti.npy')
    NUM_CLASSES = 6

    if source == 0:
        pass
    elif source == 1:
        img = np.array(Image.open(fpath))
        depth, segm = pipeline(img, hydranet, NUM_CLASSES, CMAP)
        depth = depth_to_rgb(depth)
        # f, (ax1, ax2, ax3) = plt.subplot(1, 2, figsize=(30, 20))
        # ax1.imshow(img)
        # ax1.set_title('ORIGINAL', fontsize=30)
        # ax2.imshow(segm)
        # ax2.set_title('SEGMENTATION', fontsize=30)
        # ax3.imshow(depth, cmap="plasma", vmin=0, vmax=80)
        # plt.show()
        cv2.imwrite('depth.jpg', depth)
        cv2.imwrite('segm.jpg', segm)

    elif source == 2:
        images_files = glob.glob(fpath + '/*.png')
        segm_fldr = "Segm"
        depth_fldr = "Depth"
        if os.path.exists(segm_fldr):
            shutil.rmtree(segm_fldr)
        if os.path.exists(depth_fldr):
            shutil.rmtree(depth_fldr)
        os.makedirs(segm_fldr)
        os.makedirs(depth_fldr)
        for idx, fpath in enumerate(images_files):
            fname = os.path.basename(fpath)
            img = np.array(Image.open(fpath))
            depth, segm = pipeline(img, hydranet, NUM_CLASSES, CMAP)
            cv2.imwrite("./" + depth_fldr + "/" + fname, depth)
            # cv2.imwrite("./" + segm_fldr + "/" + fname, segm)

    elif source == 3:
        video = cv2.VideoCapture(fpath)
        if video.isOpened() == False:
            print("[INFO] ERROR OPENING VIDEO STREAM OR FILE")
        output_video = []
        while video.isOpened():
            (check, frame) = video.read()
            if check:
                h, w, _ = frame.shape
                frame = np.array(frame)
                depth, segm = pipeline(frame, hydranet, NUM_CLASSES, CMAP)
                depth = depth_to_rgb(depth)
                output_video.append(np.concatenate((frame, segm, depth)))
            else:
                break
        out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (w, 3 * h))
        for i in range(len(output_video)):
            out.write(output_video[i])
        video.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hydranet: Depth, Segmentation, and Normal Heads')
    parser.add_argument('--source', type=int, required=True,
                        help='0: Camera   1: Single Image   2: Path to Multiple Images  3: Path to Video')
    parser.add_argument('--path', type=str, help='Path to the source file')
    args = parser.parse_args()

    if args.source != 0 and args.path is None:
        sys.exit("User did not specify a path ('--path' argument)")
    else:
        main(args.source, args.path)
