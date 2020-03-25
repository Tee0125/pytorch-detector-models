
PyTorch implementation of detector networks

Trained SSD300 network achevied mAP 0.617 for VOC2007-test but it is much lower than number in the original paper

Options used to train SSD300 is as in below:

* --lr 1e-3
* --use_multi_step_lr
* --milestones 600 800 1000
* --epoches 1200

In my experiences, adding augumentation improves mAP so now I am working on adding more augumentations...

# Pre-requisite

```
pip install -r requirements
```

# Train

train with default parameter

```
python detect_train.py
```

## Command Arguments
| name                | description | default |
|---------------------|-------------|:-------:|
| --model             | model name | ssd300 |
| --dataset           | dataset name | VOC |
| --dataset_root      | dataset location | downloads |
| --batch_size        | size of mini-batch | 32 |
| --lr                | learning rate for SGD | 0.001  |
| --weight_decay      | weight decay for SGD | 0.0005 |
| --gamma             | gamma for lr scheduler | 0.1 |
| --th_conf           | confidence threshold | 0.5 |
| --th_iou            | iou threshold | 0.5 |
| --resume            | resume training | None |
| --use_step_lr       | use step lr scheduler | False  |
| --step_size         | step_size for step lr scheduler | 30 |
| --use_multi_step_lr | use multi step lr scheduler | False  |
| --milestones        | milestones for multi step lr scheduler | [800, 1000, 1200] |

note: 
in case of ssd300 model, 11GB GPU memory is required for batch_size 32 and 8GB GPU memory is required for batch_size 28

## Available models
| name                    | description |
|:-----------------------:|-------------|
| ssd300                  | alias of ssd300-voc |
| ssd300-voc              | SSD with input size 300 and num_class=20 |
| ssd300-bn-voc           | batch normalization adopted version of ssd300-voc |
| ssdlite                 | alias of ssdlite-mobilenetv2-voc |
| ssdlite-mobilenetv2-voc | SSD with MobileNet v2 backbone, input size 320 and num_class=20 |

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
python detect_train.py --model ssdlite --use_multi_step_lr --milestones 500 600 700 --batch_size 25
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
| --weight            | weight file name | checkpoints/{MODEL_NAME}_latest.pth |

# Single run

```
python detect_single.py image1 [image2] [image3] [...]
```

## Command Arguments
| name                | description | default |
|---------------------|-------------|:-------:|
| --model             | model name | ssd300 |
| --weight            | weight file name | checkpoints/{MODEL_NAME}_latest.pth |
| --th_conf           | confidence threshold | 0.5 |
| --th_iou            | iou threshold | 0.5 |
| --outfile           | save result to file | None |

