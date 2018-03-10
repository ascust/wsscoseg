import cores.utils.misc as misc
import cores.utils.voc_cmap as voc_cmap
from cores.config import conf
import argparse
import os
import mxnet as mx
import numpy as np
from PIL import Image
import cPickle as pickle
from cores.data.InferenceDataProducer import InferenceDataProducer

def get_voc_mask(image_folder, output_folder, net_symbol, class_num, flist_path, epoch, model_prefix,
                 ctx, workspace, max_dim, rgb_mean, scale_list, multi_label_file=None):
    misc.my_mkdir(output_folder)
    cmap = voc_cmap.get_cmap()

    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]="0"

    seg_net = net_symbol.create_infer(class_num, workspace)
    arg_dict, aux_dict, _ = misc.load_checkpoint(model_prefix, epoch)

    if multi_label_file is not None:
        with open(multi_label_file, 'rb') as f:
            data_dict = pickle.load(f)

    mod = mx.mod.Module(seg_net, data_names=["data", "orig_data"], label_names=[], context=ctx)
    mod.bind(data_shapes=[("data", (1, 3, max_dim, max_dim)), ("orig_data", (1, 3, max_dim, max_dim))],
             for_training=False, grad_req="null")
    initializer = mx.init.Normal()
    initializer.set_verbosity(True)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=True)

    data_producer = InferenceDataProducer(
        im_root=image_folder,
        mask_root="",
        flist_path=flist_path,
        rgb_mean=rgb_mean,
        scale_list=scale_list)

    nbatch = 0

    while True:
        data = data_producer.get_data()
        if data is None:
            break
        im_list = data[0]

        label = data[1].squeeze()
        file_name = data[2]
        final_scoremaps = mx.nd.zeros((class_num, label.shape[0], label.shape[1]))

        for im in im_list:
            im, orig_size = misc.pad_image(im, 8)
            mod.reshape(data_shapes=[("data", im.shape), ("orig_data", (1, 3, orig_size[0], orig_size[1]))])
            mod.forward(mx.io.DataBatch(data=[mx.nd.array(im), mx.nd.zeros((1, 3, orig_size[0], orig_size[1]))]))

            score = mx.nd.transpose(mod.get_outputs()[0].copyto(mx.cpu()), [0, 2, 3, 1])
            score = mx.nd.reshape(score, (score.shape[1], score.shape[2], score.shape[3]))
            up_score = mx.nd.transpose(mx.image.imresize(score, label.shape[1], label.shape[0], interp=1), [2, 0, 1])
            final_scoremaps += up_score

        final_scoremaps = final_scoremaps.asnumpy()


        if multi_label_file is not None:
            tmp_label = data_dict[file_name]
            image_level_labels = np.zeros((class_num-1))
            image_level_labels[tmp_label] = 1
            image_level_labels = np.insert(image_level_labels, 0, 1)
            image_level_labels = image_level_labels.reshape((class_num, 1, 1))
            final_scoremaps *= image_level_labels

        pred_label = final_scoremaps.argmax(0)

        out_img = np.uint8(pred_label)
        out_img = Image.fromarray(out_img)
        out_img.putpalette(cmap)
        out_img.save(os.path.join(output_folder, file_name+".png"))


        nbatch += 1
        if nbatch % 10 == 0:
            print "processed %dth batch" % nbatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpu", default=0, type=int,
                        help="Device indices.")
    parser.add_argument("--epoch", default=1, type=int,
                        help="epoch number of snapshot of the init model")
    args = parser.parse_args()

    ctx = mx.gpu(args.gpu)
    model_name = "init_fcn_%s" % conf.BASE_NET
    exec ("import cores.symbols." + model_name + " as net_symbol")
    image_folder=os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER)
    misc.my_mkdir(conf.CACHE_PATH)
    output_folder=os.path.join(conf.CACHE_PATH, conf.FINAL_VOC_MASK_FOLDER)
    model_prefix = os.path.join(conf.SNAPSHOT_FOLDER, model_name)
    multi_label_file = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_MULTI_FILE)
    scale_list = conf.MULT_SCALE_EVAL

    get_voc_mask(image_folder=image_folder, output_folder=output_folder, net_symbol=net_symbol,
                 class_num=conf.CLASS_NUM, flist_path=os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_LIST),
                 epoch=args.epoch, model_prefix=model_prefix, ctx=ctx, workspace=conf.WORKSPACE,
                 max_dim=conf.MAX_INPUT_DIM, rgb_mean=conf.MEAN_RGB,
                 scale_list=scale_list, multi_label_file=multi_label_file)
