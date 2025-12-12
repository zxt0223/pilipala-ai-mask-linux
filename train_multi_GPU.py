import time
import os
import datetime
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

# [Fix] 解决多卡通信问题
os.environ["NCCL_P2P_DISABLE"] = "1"

import transforms
from my_dataset_coco import CocoDetection
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir
from torch.utils.tensorboard import SummaryWriter # [新增]

def create_model(num_classes, load_pretrain_weights=True):
    # 原版 ResNet50
    backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    return model

def main(args):
    init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # [优化] 将结果文件都放在 output_dir 下，保持整洁
    if not os.path.exists(args.output_dir):
        mkdir(args.output_dir)
        
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 结果文件路径改为绝对路径，指向 output_dir
    det_results_file = os.path.join(args.output_dir, f"det_results_{now}.txt")
    seg_results_file = os.path.join(args.output_dir, f"seg_results_{now}.txt")

    # [新增] TensorBoard (日志也放在 output_dir 下的 logs 文件夹)
    writer = None
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    print("Loading data")
    # 保持与 LEG 版本一致的增强策略，确保对比公平
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            # 如果 LEG 版加了其他增强，这里最好也加上，比如 ColorJitter
        ]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    COCO_root = args.data_path
    train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    val_dataset = CocoDetection(COCO_root, "val", data_transform["val"])

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)
        
        # 记录训练 Loss
        if writer is not None:
            writer.add_scalar('Train/Loss', mean_loss.item(), epoch)
            writer.add_scalar('Train/LR', lr, epoch)

        lr_scheduler.step()

        # 验证
        det_info, seg_info = utils.evaluate(model, data_loader_test, device=device)

        if args.rank in [-1, 0]:
            # 记录验证 mAP
            if writer is not None:
                writer.add_scalar('Val/mAP_bbox', det_info[0], epoch) # mAP 0.5:0.95
                if seg_info is not None:
                    writer.add_scalar('Val/mAP_segm', seg_info[0], epoch)

            # 写入文件 (都在 output_dir 里)
            with open(det_results_file, "a") as f:
                result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                f.write(f"epoch:{epoch} {'  '.join(result_info)}\n")

            if seg_info is not None:
                with open(seg_results_file, "a") as f:
                    result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                    f.write(f"epoch:{epoch} {'  '.join(result_info)}\n")

        if args.output_dir:
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args, 'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files, os.path.join(args.output_dir, f'model_{epoch}.pth'))

    if writer is not None: writer.close()
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='coco2017', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=26, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr-step-size', default=8, type=int)
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--print-freq', default=50, type=int)
    # [建议] 给原版一个单独的输出目录，比如 multi_train_resnet50
    parser.add_argument('--output-dir', default='wyy/pytorch-pilipala-mask', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--test-only', action="store_true")
    parser.add_argument('--world-size', default=4, type=int)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    parser.add_argument("--pretrain", type=bool, default=True)
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp") # 建议开启

    args = parser.parse_args()
    if args.output_dir: mkdir(args.output_dir)
    main(args)