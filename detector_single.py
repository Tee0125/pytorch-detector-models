import argparse
import torch
from PIL import Image

import datasets
from models import build_model, load_model
from nn import DetectPostProcess
from utils.box_util import draw_object_box
from transforms import detector_transforms as transforms


def read_image(source):
    return Image.open(source)


def prepare_input(img, size):
    t = transforms.Compose((transforms.LetterBox(),
                            transforms.Resize(size)))

    img, _ = t(img, [])

    t = transforms.Compose((transforms.ToTensor(),
                            transforms.Normalize()))

    tensor, _ = t(img, [])

    return img, tensor.unsqueeze(0)


def prepare_model(args, weight):
    # build model
    model = build_model(args, pretrained=False)
    post_process = DetectPostProcess(model.get_anchor_box(),
                                     args.th_conf,
                                     args.th_iou)

    # load weight
    load_model(model, weight)

    # transfer to GPU if possible
    if torch.cuda.is_available():
        model = model.cuda()

    return model, post_process


def inference(model, post_process, img, th_iou, th_conf):
    size = model.get_input_size()

    # prepare input
    img, x = prepare_input(img, size)

    # inference -> postprocess(softmax->nms->...)
    if torch.cuda.is_available():
        x = x.cuda()

    conf, loc = model(x)

    return img, post_process(conf, loc, th_iou, th_conf)


def single_run(model, post_process, x, dataset, th_iou, th_conf):
    # change to evaluation mode
    model.eval()

    # inference image
    img = read_image(x)

    with torch.no_grad():
        img, results = inference(model, 
                                 post_process, 
                                 img, 
                                 th_iou, 
                                 th_conf)
    
    # print results
    objs = []
    for _cls, _objs in enumerate(results[0]):
        if not _objs:
            continue
    
        label = dataset.classes[_cls]
    
        for _obj in _objs:
            _obj.append(label)
            objs.append(_obj)
    
    img = draw_object_box(img, objs)
    img.show()


def main():
    parser = argparse.ArgumentParser(description='Detector Single Test')

    parser.add_argument('inputs', type=str, nargs='*',
                        help='Input image path')
    parser.add_argument('--model', default='ssd300',
                        help='Detector model name')
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--weight', default=None,
                        help='Weight file path')
    parser.add_argument('--th_conf', default=0.5, type=float,
                        help='Confidence Threshold')
    parser.add_argument('--th_iou', default=0.5, type=float,
                        help='IOU Threshold')
    args = parser.parse_args()

    # dataset
    if args.dataset == 'VOC':
        dataset = datasets.VOCDetection
    else:
        raise Exception("unknown dataset")

    # load weight
    if args.weight:
        weight = args.weight
    else:
        weight = 'checkpoints/' + args.model + '_latest.pth'
    
    model, post_process = prepare_model(args, weight)

    for x in args.inputs:
        single_run(model, post_process, x, dataset)


if __name__ == "__main__":
    main()

