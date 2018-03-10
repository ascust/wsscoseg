from easydict import EasyDict as EDict

conf = EDict()

conf.CLASS_NUM = 21 # in this case voc dataset.
conf.MEAN_RGB = (123, 117, 104) #RGB not BGR
conf.WD = 5e-4
conf.MOMENTUM = 0.9
conf.WORKSPACE = 512
conf.DOWN_SAMPLE_SCALE = 8

#train init model
conf.LR_INIT = 16e-4
conf.EPOCH_SIZE_INIT = 200
conf.MAX_EPOCH_INIT = 40
conf.BATCH_SIZE_INIT = 16
conf.CROP_SIZE_INIT = 320
conf.SCALE_RANGE_INIT = [0.7, 1.3]

#train final model
conf.LR_FINAL = 16e-4
conf.EPOCH_SIZE_FINAL = 700
conf.MAX_EPOCH_FINAL = 40
conf.BATCH_SIZE_FINAL = 16
conf.CROP_SIZE_FINAL = 320
conf.SCALE_RANGE_FINAL = [0.7, 1.3]

#for evaluate init and final models
conf.CPU_WORKER_NUM = 8

conf.EVAL_WAIT_TIME = 0.3 # hour
conf.MAX_INPUT_DIM = 800
conf.MULT_SCALE_EVAL = [0.7, 1.0, 1.3] #multi scale evaluation can be used

conf.CRF_POS_XY_STD = 2
conf.CRF_POS_W = 3
conf.CRF_BI_RGB_STD = 3
conf.CRF_BI_XY_STD = 55
conf.CRF_BI_W = 4

#for dataset
#SBD_PATH is the one named "dataset", which has "cls", "img", "inst", "train.txt" and "val.txt".
#please download at https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
#VOCDEVKIT_PATH is the one named "VOC2012", which has "Annotations", "ImageSets", "JPEGImages", "SegmentationClass",
conf.DATASET_PATH = "dataset"
conf.SBD_PATH = "dataset/benchmark_RELEASE/dataset"
conf.VOCDEVKIT_PATH = "dataset/VOCdevkit/VOC2012"
conf.VOC_TRAIN_MULTI_FILE = "voc_multi_file.p"
conf.VOC_TRAIN_IM_FOLDER = "train_images"
conf.VOC_VAL_IM_FOLDER = "val_images"
conf.VOC_VAL_MASK_FOLDER = "val_masks"
conf.VOC_TRAIN_LIST = "train_list.txt"
conf.VOC_VAL_LIST = "val_list.txt"
conf.FINAL_VOC_MASK_FOLDER = "final_voc_masks"
conf.CACHE_PATH = "cache"
conf.BASE_NET = "resnet50"
conf.WEB_IMAGE_FOLDER = "web_images"
conf.WEB_IMAGE_MASKS_FOLDER = "web_labels"
conf.WEB_TRAIN_LIST = "web_train_list.txt"
conf.LOG_FOLDER = "log"
conf.SNAPSHOT_FOLDER = "snapshots"
conf.OUTPUT_FOLDER = "outputs"



