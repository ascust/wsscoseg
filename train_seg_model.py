import argparse
from cores.seg_wrapper import train_seg_wrapper
import os
import mxnet as mx
import logging
from cores.config import conf
import cores.utils.misc as misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpus", default="0",
                        help="Device indices.")
    parser.add_argument("--epoch", default=0, type=int,
                        help="Starting epoch.")
    parser.add_argument("--lr", default=-1, type=float,
                        help="Learning rate.")
    parser.add_argument("--model", default="init",
                        help="train the init model or final model. either \"init\" or \"final\"")

    args = parser.parse_args()
    misc.my_mkdir(conf.SNAPSHOT_FOLDER)
    misc.my_mkdir(conf.LOG_FOLDER)
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(conf.CPU_WORKER_NUM)

    conf.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    assert args.model in ["init", "final"], "wrong model type. Should be either \"init\" or \"final\""
    log_file_name = os.path.join(conf.LOG_FOLDER, "train_%s_model.log"%args.model)
    if os.path.exists(log_file_name) and args.epoch==0:
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)


    if args.model == "init":
        conf.model_name = "init_fcn_%s" % conf.BASE_NET
        exec ("import cores.symbols." + conf.model_name + " as net_symbol")

        if args.lr == -1:
            conf.lr = conf.LR_INIT
        else:
            conf.lr = args.lr
        conf.mask_root = os.path.join(conf.DATASET_PATH, conf.WEB_IMAGE_MASKS_FOLDER)
        conf.im_root = os.path.join(conf.DATASET_PATH, conf.WEB_IMAGE_FOLDER)
        conf.flist_path = os.path.join(conf.DATASET_PATH, conf.WEB_TRAIN_LIST)
        conf.crop_size = conf.CROP_SIZE_INIT
        conf.scale_range = conf.SCALE_RANGE_INIT
        conf.label_shrink_scale = 1.0 / conf.DOWN_SAMPLE_SCALE
        conf.epoch_size = conf.EPOCH_SIZE_INIT
        conf.max_epoch = conf.MAX_EPOCH_INIT
        conf.batch_size = conf.BATCH_SIZE_INIT
        conf.use_g_labels=False

    else:
        conf.model_name = "final_fcn_%s" % conf.BASE_NET
        exec ("import cores.symbols." + conf.model_name + " as net_symbol")
        conf.mask_root = os.path.join(conf.CACHE_PATH, conf.FINAL_VOC_MASK_FOLDER)
        conf.im_root = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER)
        if args.lr == -1:
            conf.lr = conf.LR_FINAL
        else:
            conf.lr = args.lr
        conf.flist_path = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_LIST)
        conf.crop_size = conf.CROP_SIZE_FINAL
        conf.scale_range = conf.SCALE_RANGE_FINAL
        conf.label_shrink_scale = 1.0 / conf.DOWN_SAMPLE_SCALE
        conf.epoch_size = conf.EPOCH_SIZE_FINAL
        conf.max_epoch = conf.MAX_EPOCH_FINAL
        conf.batch_size = conf.BATCH_SIZE_FINAL
        conf.use_g_labels=True

    conf.model_prefix = os.path.join(conf.SNAPSHOT_FOLDER, conf.model_name)
    conf.epoch = args.epoch
    conf.init_weight_file = "models/%s.params" % conf.BASE_NET


    logging.info(conf)
    logging.info("start training the %s model" % args.model)

    train_seg_wrapper(ctx=conf.ctx, epoch=conf.epoch, lr=conf.lr,
                      model_prefix=conf.model_prefix, symbol=net_symbol, class_num=conf.CLASS_NUM,
                      workspace=conf.WORKSPACE, init_weight_file=conf.init_weight_file,
                      im_root=conf.im_root, mask_root=conf.mask_root, flist_path=conf.flist_path,
                      use_g_labels=conf.use_g_labels, rgb_mean=conf.MEAN_RGB, crop_size=conf.crop_size,
                      scale_range=conf.scale_range, label_shrink_scale=conf.label_shrink_scale,
                      epoch_size=conf.epoch_size, max_epoch=conf.max_epoch,
                      batch_size=conf.batch_size, wd=conf.WD, momentum=conf.MOMENTUM)

