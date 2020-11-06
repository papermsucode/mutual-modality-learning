# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.distributed as distr

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
import datetime

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    distr.init_process_group(backend='nccl',init_method=args.init_method,
                             rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(hours=1.))

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)

    args.modality = args.modality.split(',')
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality[args.rank], full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense{}'.format(args.dense_length)
    elif args.random_sample:
        args.store_name += '_random{}'.format(args.dense_length)
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    if len(args.modality)>1:
        args.store_name += '_ML{}'.format(args.rank)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality[args.rank],
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda(args.gpus[0])

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)

        if args.modality[args.rank] not in args.tune_from or (args.modality[args.rank]=='RGB' and 'RGBDiff' in args.tune_from):
            if 'Flow' in args.tune_from:
                model._construct_flow_model(model.base_model)
            elif 'RGBDiff' in args.tune_from:
                model._construct_diff_model(model.base_model)
            else:
                model._construct_rgb_model(model.base_model)
            model.load_state_dict(model_dict)
            if args.modality[args.rank]=='Flow':
                model._construct_flow_model(model.base_model)
            elif args.modality[args.rank]=='RGBDiff':
                model._construct_diff_model(model.base_model)
            else:
                model._construct_rgb_model(model.base_model)
        else:
            model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    train_loader = None
    if args.rank==0:
        input_mean = []
        input_std = []
        data_length = []
        for moda in args.modality:
            if moda=='RGB':
                input_mean += [0.485, 0.456, 0.406]
                input_std += [0.229, 0.224, 0.225]
                data_length += [1]
            elif moda=='Flow':
                input_mean += [0.5]*10
                input_std += [0.226]*10
                data_length += [5]
            elif moda=='RGBDiff':
                input_mean += [0.]*18
                input_std += [1.]*18
                data_length += [6]

        normalize = GroupNormalize(input_mean, input_std)
        train_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                       modality=args.modality,
                       new_length=data_length,
                       image_tmpl=prefix,
                       transform=torchvision.transforms.Compose([
                           train_augmentation,
                           Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize,
                       ]), dense_sample=args.dense_sample, random_sample=args.random_sample,
                       dense_length=args.dense_length),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=True)  # prevent something not % n_GPU

    if args.modality[args.rank]=='RGB':
        normalize_val = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif args.modality[args.rank]=='Flow':
        normalize_val = GroupNormalize([0.5]*10, [0.226]*10)
    elif args.modality[args.rank]=='RGBDiff':
        normalize_val = IdentityTransform()

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet([args.root_path[args.rank]], [args.val_list[args.rank]], num_segments=args.num_segments,
                   new_length=[data_length[args.rank]],
                   modality=[args.modality[args.rank]],
                   image_tmpl=[prefix[args.rank]],
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize_val,
                   ]), dense_sample=args.dense_sample, dense_length=args.dense_length),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpus[-1])
    else:
        raise ValueError("Unknown loss type")

    if len(args.modality)>1:
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda(args.gpus[-1])
        logsoftmax = torch.nn.LogSoftmax(dim=1).cuda(args.gpus[-1])
        softmax = torch.nn.Softmax(dim=1).cuda(args.gpus[-1])
    else:
        kl_loss = None
        logsoftmax = None
        softmax = None

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, kl_loss, logsoftmax, softmax, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, kl_loss, logsoftmax, softmax, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_kl = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total = 0
    shift = 0
    for i,moda in enumerate(args.modality):
        tmp = total
        if moda=='RGB':
            total += 3
        elif moda=='Flow':
            total += 10
        elif moda=='RGBDiff':
            total += 18
        if i==0:
            shift = total
        if i==args.rank and i>0:
            start_ind = tmp-shift
            end_ind = total-shift
        elif i==args.rank and i==0:
            start_ind = 0
            end_ind = total

    if args.rank==0:
        inds = []
        for x in range(args.num_segments):
            inds.extend(list(range(x*total+start_ind,x*total+end_ind)))
        send_inds = []
        for x in range(args.num_segments):
            send_inds.extend(list(range(x*total+end_ind,x*total+total)))
    else:
        inds = []
        for x in range(args.num_segments):
            inds.extend(list(range(x*(total-shift)+start_ind,x*(total-shift)+end_ind)))

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode5r
    model.train()

    if args.rank==0:
        iter_through = train_loader
    else:
        iter_through = range(int(len([x for x in open(args.train_list[0])])/args.batch_size))

    end = time.time()
    for i, data in enumerate(iter_through):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.rank==0:
            input, target = data

            target = target.cuda(args.gpus[-1])
            input = input.cuda(args.gpus[0])

            if args.world_size>1:
                torch.distributed.broadcast(input[:,send_inds].contiguous(),0)
                torch.distributed.broadcast(target,0)
        else:
            input = torch.zeros((args.batch_size,(total-shift)*args.num_segments,224,224)).cuda(args.gpus[0])
            target = torch.zeros((args.batch_size,),dtype=torch.int64).cuda(args.gpus[-1])
            torch.distributed.broadcast(input,0)
            torch.distributed.broadcast(target,0)

        input_var = torch.autograd.Variable(input[:,inds].contiguous())
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var).cuda(args.gpus[-1])
        loss1 = criterion(output, target_var)

        if args.world_size>1:
            reduce_output = output.clone().detach()
            distr.all_reduce(reduce_output)
            reduce_output = (reduce_output-output.detach())/(args.world_size-1)
            loss2 = kl_loss(logsoftmax(output), softmax(reduce_output.detach()))
        else:
            loss2 = torch.tensor(0.)
        loss = loss1+loss2

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss1.item(), input.size(0))
        loss_kl.update(loss2.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'LossKL {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(iter_through), batch_time=batch_time,
                data_time=data_time, loss1=losses, loss2=loss_kl, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('loss/mutual', loss_kl.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(args.gpus[-1])

            # compute output
            output = model(input).cuda(args.gpus[-1])
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
