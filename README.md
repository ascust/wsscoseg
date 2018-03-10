# Weakly Supervised Semantic Segmentation Based on Web Image Co-segmentation

Authors: Tong Shen, Guosheng Lin, Lingqiao Liu, Chunhua Shen, Ian Reid

## Abstract

Training a Fully Convolutional Network (FCN) for semantic segmentation requires a large number of masks with pixel level 
labelling, which involves a large amount of human labour and time for annotation. In contrast, web images and their image-level 
labels are much easier and cheaper to obtain. In this work, we propose a novel method for weakly supervised semantic segmentation 
with only image-level labels. The method utilizes the internet to retrieve a large number of images and uses a large scale 
co-segmentation framework to generate masks for the retrieved images. We first retrieve images from search engines, 
e.g. Flickr and Google, using semantic class names as queries, e.g. class names in the dataset PASCAL VOC 2012. We then 
use high quality masks produced by co-segmentation on the retrieved images as well as the target dataset images with image 
level labels to train segmentation networks. We obtain an IoU score of 56.9 on test set of PASCAL VOC 2012, which reaches 
the state-of-the-art performance.

![Overview](http://gdurl.com/HtYD)

## Citing the paper

Please consider citing us if you find it useful:

        @inproceedings{Shen:2017:wss,
          author    = {Tong Shen and
                       Guosheng Lin and
                       Lingqiao Liu and
                       Chunhua Shen and
                       Ian Reid},
          title     = {Weakly Supervised Semantic Segmentation Based on Web Image Co-segmentation},
          booktitle = {BMVC},
          year      = {2017}
        }

## Dependencies
The code is implemented in MXNet. Please go to the official website ([HERE](https://mxnet.apache.org)) for installation.
Please make sure the MXNet is compiled with OpenCV support. 

The other python dependences can be found in "dependencies.txt" and can be installed:

```pip install -r dependencies.txt```

## Dataset

### Web data

The Web data can be downloaded [here](https://cloudstor.aarnet.edu.au/plus/index.php/s/rIrbWH1hn0Rm52H). Since the
co-segmentation code is not included ([Original Github](https://github.com/endernewton/subdiscover)), one can either
run the code to get the masks or use the masks provided, which are already processed. To use the provided masks, extract
the files and put all the images and masks in "_dataset/web_images_" and "_dataset/web_labels_" respectively. No subfolders
should be used.

### PACAL VOC data
For PASCAL VOC data, please download PASCAL VOC12 ([HERE](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) and
 SBD ([HERE](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tg)). Then
 extract the files into folder "_dataset_" and run:
 
 ```python create_dataset.py```
 
## Training
First download the Resnet50 model pretrained on ImageNet ([HERE](https://1drv.ms/u/s!ArsE1Wwv6I6dgQl_JrthaxDCMi-Q)).
Put it in folder "_models_".
### Training the Initial Mask Generator

To train the "Initial Mask Generator", simply run:

```python train_seg_model.py --model init --gpus 0,1,2,3```

To evaluate a certain snapshot (for example epoch X), run:

```python eval_seg_model.py --model init --gpu 0 --epoch X```

To evaluate all the snapshots, run:

```python eval_loop.py --model init --gpu 0```

The evaluated snapshots will have a corresponding folder in "_outputs_". This ```eval_loop.py``` will check if there is 
any unevaluated snapshots and evaluate them.

To further improve the score, finetune a snapshot (for example epoch X) with smaller learning rate:

```python train_seg_model.py --model init --gpus 0,1,2,3 --epoch X --lr 16e-5```

### Training the Final Model

Check the evaluation log in "_log/eval_model.log_" and find the best snapshot 
(Download a trained one [HERE](https://1drv.ms/u/s!ArsE1Wwv6I6dgQcyRsOver3xkMWG)) 
for the mask generator. For example the best epoch is "X", then run:

```
python est_voc_train_masks.py --gpu 0 --epoch X
python train_seg_model.py --model final --gpus 0,1,2,3
python eval_loop.py --model final --gpu 0
```

The above code will estimate the masks for the VOC training images and train the final model. 


## Evaluation

The snapshots will be saved in folder "_snapshots_". To evaluate a snapshot, simply use (for example epoch X):

```python eval_seg_model.py --model final --gpu 0 --epoch X```

There are other flags:

```
--ms                use multi-scale for inference
--savemask          save output masks
--crf               use CRF as postprocessing
```

There is a trained model that can be downloaded [HERE](https://1drv.ms/u/s!ArsE1Wwv6I6dgQj1ueNm8cs_A-gw).

Download the model and put it in folder "_snapshots_". Run:

```python eval_seg_model.py --model final --gpu 0 --epoch 23 --crf --ms```

It will get IoU of 56.4, as reported in the paper.

## Demo

A demo code is given in "_demo_". Download the final model 
([HERE](https://1drv.ms/u/s!ArsE1Wwv6I6dgQj1ueNm8cs_A-gw)) and put it in the folder "_snapshots_".
Please use Jupyter to run "_Demo.ipynb_". 

## Examples

There are some examples here.

![examples](http://gdurl.com/nzGw)