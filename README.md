
PyTorch implementation of:

* [SSD: Single Shot Multibox Detector](https://arxiv.org/abs/1512.02325)
* [SSD-Lite: Lightweight SSD based on MobileNet](https://arxiv.org/abs/1801.04381) 
* [RetinaNet: Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

# Evaluation Results on VOC2007 test dataset

| model      | augumentation | training set          | epochs | loss  | mAP    | checkpoint | 
|:----------:|:-------------:|:---------------------:|:------:|:-----:|:------:|:----------:|
| ssd300     | X             | VOC2007 trainval      | 200    | 0.081 | 0.430  |            |
| ssd300     | X             | VOC2007/2012 trainval | 200    | 0.081 | 0.522  |            |
| ssdlite    | O             | VOC2007/2012 trainval | 200    | 2.469 | 0.712  | [download](https://drive.google.com/file/d/1QplZSBF3-ZiEDUlVFJsXMcwYlo_q3gzK) |
| ssd300     | O             | VOC2007/2012 trainval | 200    | 2.139 | 0.776  | [download](https://drive.google.com/open?id=1waoa_BHAPNFpCZc86tfOCqU-aIMwBCQ3) |
| ssd512     | O             | VOC2007/2012 trainval | 200    | 2.012 | 0.792  | [download](https://drive.google.com/file/d/1-Nw730Uf5lJ_otRbJpTSwg0OKqeTM2Sq) |

option used for training is `--use_multi_step_lr`

# Status

- [x] Implement SSD / SSDLite model
- [x] Train SSD / SSDLite and add evaluation results
- [x] Implement RetinaNet
- [ ] Train RetinaNet with Focal Loss and add evaluataion result
- [ ] Implement EfficientDet
- [ ] Train EfficientDet and add evaluation result
- [ ] Support COCO dataset
- [ ] Support custom dataset
- [ ] Implement VGG challenge's mAP calculator
- [ ] Implement COCO challenge's mAP calculator

# Pre-requisite

```
pip install -r requirements.txt
```

# Train

train with default parameter (dataset will be downloaded automatically)

```
python detect_train.py
```

## Command Arguments
| name                | description | default |
|---------------------|-------------|:-------:|
| --model             | model name | ssd300 |
| --dataset           | dataset name | VOC |
| --dataset_root      | dataset location | downloads |
| --download          | download dataset | False |
| --epochs            | number of epochs to run | 200 |
| --batch_size        | size of mini-batch | 32 |
| --lr                | learning rate for SGD | 1e-3  |
| --weight_decay      | weight decay for SGD | 5e-4 |
| --gamma             | gamma for lr scheduler | 0.1 |
| --th_conf           | confidence threshold | 0.5 |
| --th_iou            | iou threshold | 0.5 |
| --resume            | resume training | None |
| --use_step_lr       | use step lr scheduler | False  |
| --step_size         | step_size for step lr scheduler | 30 |
| --use_multi_step_lr | use multi step lr scheduler | False  |
| --use_plateau_lr    | use plateau lr scheduler | False  |
| --milestones        | milestones for multi step lr scheduler | 140 170 |
| --disable_augmentation | disable random augmentation | False |
| --enable_letterbox  | enable letter boxing image | False |

note: 
in case of ssd300 model, 11GB GPU memory is required for batch_size 32 and 8GB GPU memory is required for batch_size 28

## Available models
| name                    | description |
|:-----------------------:|-------------|
| ssd300                  | alias of `ssd300-voc` |
| ssd300-voc              | SSD with input size 300x300 and num_class=20 |
| ssd300-bn-voc           | batch normalization adopted version of `ssd300-voc` |
| ssd512                  | alias of `ssd512-voc` |
| ssd512-voc              | SSD with input size 512x512 and num_class=20 |
| ssdlite                 | alias of `ssdlite-mobilenetv2-voc` |
| ssdlite-mobilenetv2-voc | SSD with MobileNet v2 backbone, input size 320x320 and `num_class`=20 |
| retinanet               | alias of `retinanet-50-500-voc` |
| retinanet-50-500-voc    | RetinaNet with resenet-50 backbone, input size 500x500 and `num_class`=20 |
| retinanet-101-500-voc   | RetinaNet with resenet-101 backbone, input size 500x500 and `num_class`=20  |
| retinanet-50-600-voc    | RetinaNet with resenet-50 backbone, input size 600x600 and `num_class`=20  |
| retinanet-101-600-voc   | RetinaNet with resenet-101 backbone, input size 600x600 and `num_class`=20  |

## Available datasets
| name                 | description |
|:--------------------:|-------------|
| VOC                  | VOC dataset (2007+2012) |
| VOC2007              | VOC dataset (2007 only) |
| VOC2012              | VOC dataset (2012 only) |

## Example

Training SSD with multi step lr and batch_size is 32 (default)

```
python detect_train.py --use_multi_step_lr
```

Training SSD-Lite with multi step lr and batch_size 25 (SSD-Lite model is not tested yet)

```
python detect_train.py --model ssdlite --use_multi_step_lr --milestones 140 160 --batch_size 25
```

Resume training

```
python detect_train.py --resume checkpoints/ssdlite_latest.pth
```

# Evaluation

calculate mAP with test image set

```
python detect_eval.py
```

## Command Arguments
| name                | description  | default |
|---------------------|--------------|:-------:|
| --model             | model name   | ssd300  |
| --dataset           | dataset name | VOC     |
| --dataset_root      | dataset location | downloads |
| --weight            | weight file name | `checkpoints/{MODEL_NAME}_latest.pth` |
| --enable_letterbox  | enable letter boxing image | False |

# Single run

```
python detect_single.py image1 [image2] [image3] [...]
```

## Command Arguments
| name                | description | default |
|---------------------|-------------|:-------:|
| --model             | model name | ssd300 |
| --weight            | weight file name | `checkpoints/{MODEL_NAME}_latest.pth` |
| --th_conf           | confidence threshold | 0.5 |
| --th_iou            | iou threshold | 0.5 |
| --enable_letterbox  | enable letter boxing image | False |
| --outfile           | save result to file | None |

