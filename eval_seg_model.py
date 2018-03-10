import argparse
from cores.seg_wrapper import test_seg_wrapper
import os
import mxnet as mx
import logging
from cores.config import conf
import cores.utils.misc as misc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpu", default="0",
                        help="Device index.")
    parser.add_argument("--epoch", default=0, type=int,
                        help="snapshot name for evaluation")
    parser.add_argument("--savemask", help="whether save the prediction masks.",
                        action="store_true")
    parser.add_argument("--savescoremap", help="whether save the prediction scoremaps.",
                        action="store_true")
    parser.add_argument("--model", default="init",
                        help="evaluate the init model or final model. either \"init\" or \"final\"")
    parser.add_argument("--crf", help="whether use crf for post processing.",
                        action="store_true")
    parser.add_argument("--ms", help="whether use multi-scale for inference.",
                        action="store_true")
    args = parser.parse_args()

    ctx = [mx.gpu(int(args.gpu))]
    misc.my_mkdir(conf.OUTPUT_FOLDER)
    misc.my_mkdir(conf.LOG_FOLDER)

    log_file_name = os.path.join(conf.LOG_FOLDER, "eval_model.log")
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    assert args.model == "init" or args.model == "final"

    if args.model == "init":
        conf.model_name = "init_fcn_%s" % conf.BASE_NET
        exec("import cores.symbols."+ conf.model_name +" as net_symbol")
    else:
        conf.model_name = "final_fcn_%s" % conf.BASE_NET
        exec("import cores.symbols."+ conf.model_name +" as net_symbol")

    if args.ms:
        conf.scale_list = conf.MULT_SCALE_EVAL
    else:
        conf.scale_list = [1.0]

    conf.epoch = args.epoch
    conf.save_mask = args.savemask
    conf.save_scoremap = args.savescoremap
    conf.im_root = os.path.join(conf.DATASET_PATH, conf.VOC_VAL_IM_FOLDER)
    conf.mask_root = os.path.join(conf.DATASET_PATH, conf.VOC_VAL_MASK_FOLDER)
    conf.flist_path = os.path.join(conf.DATASET_PATH, conf.VOC_VAL_LIST)


    conf.use_crf = args.crf
    crf_params = {}
    crf_params["pos_xy_std"] = conf.CRF_POS_XY_STD
    crf_params["pos_w"] = conf.CRF_POS_W
    crf_params["bi_xy_std"] = conf.CRF_BI_XY_STD
    crf_params["bi_rgb_std"] = conf.CRF_BI_RGB_STD
    crf_params["bi_w"] = conf.CRF_BI_W

    logging.info(conf)

    test_seg_wrapper(epoch=conf.epoch, ctx=ctx, output_folder=conf.OUTPUT_FOLDER, model_name=conf.model_name,
                     save_mask=conf.save_mask, save_scoremap=conf.save_scoremap, net_symbol=net_symbol,
                     class_num=conf.CLASS_NUM, workspace=conf.WORKSPACE,
                     snapshot_folder=conf.SNAPSHOT_FOLDER, max_dim=conf.MAX_INPUT_DIM,
                     im_root=conf.im_root, mask_root=conf.mask_root, flist_path=conf.flist_path,
                     rgb_mean=conf.MEAN_RGB, scale_list=conf.scale_list, class_names=misc.get_voc_class_names(),
                     use_crf=conf.use_crf, crf_params=crf_params)




