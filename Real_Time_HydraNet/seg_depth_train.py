import torch
import argparse
import sys
import glob
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataloader import preprocess
from torch.utils.data import DataLoader
from utils.dataloader import JointDataset
from utils.hydra import Encoder, Decoder
from utils.hydra import InvHuberLoss
from model_helpers import Saver, load_state_dict
import operator
import json
import logging
from utils.hydra import AverageMeter
from tqdm import tqdm
from utils.hydra import MeanIoU, RMSE


def train(model, opts, crits, dataloader, loss_coeffs=(1.0,), grad_norm=0.0):
    model.train()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        input = sample["image"].float().to(device)
        targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]
        # [[sample["depth"].to(device), sample["segm"].to(device)]
        # input, targets = get_input_and_targets(sample=sample, dataloader=dataloader, device=device) # Get the data
        outputs = model(input)  # Forward
        # outputs = list(outputs)

        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            loss += loss_coeff * crit(
                F.interpolate(
                    out, size=target.size()[1:], mode="bilinear", align_corners=False
                ).squeeze(dim=1),
                target.squeeze(dim=1),
            )

        # Backward
        for opt in opts:
            opt.zero_grad()
        loss.backward()
        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        for opt in opts:
            opt.step()

        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )


def validate(model, metrics, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model.eval()
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]

            # input, targets = get_input_and_targets(sample=sample, dataloader=dataloader, device=device)
            targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]

            # Forward
            outputs = model(input)
            # outputs = make_list(outputs)

            # Backward
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
                        .squeeze(dim=1)
                        .cpu()
                        .numpy(),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals


def main(images, depth, seg):
    img_path = sorted(glob.glob(images + '/*.png'))
    dep_path = sorted(glob.glob(depth + '/*.png'))
    seg_path = sorted(glob.glob(seg + '/*.png'))
    train_file = (img_path, dep_path, seg_path)
    train_batch_size = 4
    val_batch_size = 4

    CMAP = np.load('cmap_nyud.npy')

    transform_train, transform_val = preprocess(img_path, seg, dep_path)

    trainloader = DataLoader(
        JointDataset(train_file, transform=transform_train, ),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # TODO: WORK ON SPLITTING DATASET FOR TRAIN/VAL
    valloader = DataLoader(
        JointDataset(val_file, transform=transform_val, ),
        batch_size=val_batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True,
        drop_last=False, )

    encoder = Encoder()
    encoder.load_state_dict(torch.load("mobilenetv2-e6e8dd43.pth"))
    num_classes = (40, 1)
    decoder = Decoder(encoder._out_c, num_classes)

    ignore_index = 255
    ignore_depth = 0

    crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
    crit_depth = InvHuberLoss(ignore_index=ignore_depth).cuda()

    lr_encoder = 1e-2
    lr_decoder = 1e-3
    momentum_encoder = 0.9
    momentum_decoder = 0.9
    weight_decay_encoder = 1e-5
    weight_decay_decoder = 1e-5

    optims = [torch.optim.SGD(encoder.parameters(), lr=lr_encoder, momentum=momentum_encoder,
                              weight_decay=weight_decay_encoder),
              torch.optim.SGD(decoder.parameters(), lr=lr_decoder, momentum=momentum_decoder,
                              weight_decay=weight_decay_decoder)]

    n_epochs = 1000

    init_vals = (0.0, 10000.0)
    comp_fns = [operator.gt, operator.lt]
    ckpt_dir = "./"
    ckpt_path = "./checkpoint.pth.tar"

    saver = Saver(
        args=locals(),
        ckpt_dir=ckpt_dir,
        best_val=init_vals,
        condition=comp_fns,
        save_several_mode=all,
    )

    hydranet = nn.DataParallel(nn.Sequential(encoder, decoder).cuda())  # Use .cpu() if you prefer a slow death

    print("Model has {} parameters".format(sum([p.numel() for p in hydranet.parameters()])))

    start_epoch, _, state_dict = saver.maybe_load(ckpt_path=ckpt_path,
                                                  keys_to_load=["epoch", "best_val", "state_dict"], )
    load_state_dict(hydranet, state_dict)

    if start_epoch is None:
        start_epoch = 0

    print(start_epoch)

    opt_scheds = []
    for opt in optims:
        opt_scheds.append(
            torch.optim.lr_scheduler.MultiStepLR(opt, np.arange(start_epoch + 1, n_epochs, 100), gamma=0.1))

    for i in range(start_epoch, n_epochs):
        for sched in opt_scheds:
            sched.step(i)
        hydranet.train()  # Set to train mode
        train(...)  # Call the train function

        if i % val_every == 0:
            model1.eval()  # Set to Eval Mode
            with torch.no_grad():
                vals = validate(...)  # Call the validate function

    crop_size = 400
    batch_size = 4
    val_batch_size = 4
    val_every = 5
    loss_coeffs = (0.5, 0.5)

    for i in range(start_epoch, n_epochs):
        for sched in opt_scheds:
            sched.step(i)

        print("Epoch {:d}".format(i))
        train(hydranet, optims, [crit_segm, crit_depth], trainloader, loss_coeffs)

        if i % val_every == 0:
            metrics = [MeanIoU(num_classes[0]), RMSE(ignore_val=ignore_depth), ]

            with torch.no_grad():
                vals = validate(hydranet, metrics, valloader)
            saver.maybe_save(new_val=vals, dict_to_save={"state_dict": hydranet.state_dict(), "epoch": i})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hydranet: Depth and Segmentation Training')
    parser.add_argument('--images', type=str, required=True,
                        help='path to images directory')
    parser.add_argument('--depth', type=str, required=True,
                        help='path to depth directory')
    parser.add_argument('--seg', type=str, required=True,
                        help='path to seg directory')
    args = parser.parse_args()

    if args.source != 0 and args.path is None:
        sys.exit("User did not specify a path ('--path' argument)")
    else:
        main(args.images, args.depth, args.seg)
