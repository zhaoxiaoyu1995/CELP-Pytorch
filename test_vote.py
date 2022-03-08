import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from CELP_PFENet.model.PFENet import CELPNet
from CELP_PFENet.util import dataset, transform, config
from CELP_PFENet.util.util import AverageMeter, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='./CELP_PFENet/config/pascal/pascal_split0_resnet50.yaml',
                        help='config file')
    parser.add_argument('opts', help='see CELP_PFENet/config/pascal/pascal_split0_resnet50.yaml for all options',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def set_logger(output_dir):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log')
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main():
    global args, writer
    args = get_parser()
    writer = SummaryWriter(args.save_path)
    set_logger(args.save_path)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    torch.cuda.set_device(args.train_gpu)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = CELPNet(layers=args.layers, classes=2, zoom_factor=8,
                    criterion=nn.CrossEntropyLoss(ignore_index=255),
                    pretrained=True, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg)

    logging.info("=> creating model ...")
    logging.info("Classes: {}".format(args.classes))
    logging.info(model)
    print(args)

    model = model.cuda()

    if args.weight:
        if os.path.isfile(args.weight):
            logging.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded weight '{}'".format(args.weight))
        else:
            logging.info("=> no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                               data_list=args.val_list, transform=val_transform, mode='val',
                               use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    logging.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter_list = [AverageMeter() for i in range(5)]
    intersection_meter_list = [AverageMeter() for i in range(5)]
    union_meter_list = [AverageMeter() for i in range(5)]
    target_meter_list = [AverageMeter() for i in range(5)]
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5
    class_intersection_meter_dict = {}
    class_union_meter_dict = {}
    for i in range(5):
        class_intersection_meter_dict[str(i)] = [0] * split_gap
        class_union_meter_dict[str(i)] = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    if args.split != 999:
        if args.use_coco:
            test_num = 20000
        else:
            test_num = 5000
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    for e in range(20):
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if (iter_num - 1) * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            start_time = time.time()
            with torch.no_grad():
                model.shot = 1
                output = model(s_x=s_input.permute(1, 0, 2, 3, 4), s_y=s_mask.permute(1, 0, 2, 3),
                               x=input.repeat(5, 1, 1, 1), y=target.repeat(5, 1, 1))

            model_time.update(time.time() - start_time)
            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            loss = criterion(torch.mean(output, dim=0, keepdim=True), target)
            loss = torch.mean(loss)

            output_temp = output.clone()
            target_temp = target.clone()
            subcls_temp = [subcls[i].clone() for i in range(len(subcls))]
            for i_k in range(5):
                loss_meter = loss_meter_list[i_k]
                intersection_meter = intersection_meter_list[i_k]
                union_meter = union_meter_list[i_k]
                target_meter = target_meter_list[i_k]
                class_intersection_meter = class_intersection_meter_dict[str(i_k)]
                class_union_meter = class_union_meter_dict[str(i_k)]

                output = output_temp.max(1)[1]
                output = torch.sum(output, dim=0, keepdim=True)
                output = (output >= (i_k + 1)).float()

                intersection, union, new_target = intersectionAndUnionGPU(output, target_temp, args.classes,
                                                                          args.ignore_label)
                intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target_temp.cpu().numpy(), new_target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

                subcls = subcls_temp[0].cpu().numpy()[0]
                class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
                class_union_meter[(subcls - 1) % split_gap] += union[1]

                accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
                loss_meter.update(loss.item(), input.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                if ((i + 1) % (test_num / 100) == 0):
                    logging.info('Test: [{}/{}] '
                                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                                 'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                                 'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                                   data_time=data_time,
                                                                   batch_time=batch_time,
                                                                   loss_meter=loss_meter,
                                                                   accuracy=accuracy))

    for k in range(5):
        loss_meter = loss_meter_list[k]
        intersection_meter = intersection_meter_list[k]
        union_meter = union_meter_list[k]
        target_meter = target_meter_list[k]
        class_intersection_meter = class_intersection_meter_dict[str(k)]
        class_union_meter = class_union_meter_dict[str(k)]

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        class_iou_class = []
        class_miou = 0
        for i in range(len(class_intersection_meter)):
            class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
            class_iou_class.append(class_iou)
            class_miou += class_iou
        class_miou = class_miou * 1.0 / len(class_intersection_meter)
        logging.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
        for i in range(split_gap):
            logging.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))

        logging.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logging.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logging.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
