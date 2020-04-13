import argparse
import torch

from models import build_model, load_model
from helpers import DetectionEvaluator

def prepare_model(args):
    model = build_model(args, pretrained=False)

    load_model(model, args.weight)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def main():
    parser = argparse.ArgumentParser(description='Detector Validation')

    parser.add_argument('--model', default='ssd300',
                        help='Detector model name')
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default='downloads',
                        help='Dataset root directory path')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--weight', default=None,
                        help='Weight file path')
    parser.add_argument('--th_iou', default=0.5, type=float,
                        help='IOU Threshold')
    parser.add_argument('--th_conf', default=0.05, type=float,
                        help='Confidence Threshold')
    parser.add_argument('--enable_letterbox', default=False, 
                        action='store_true',
                        help='Enable letterboxing image')
    args = parser.parse_args()


    # load weight
    if not args.weight:
        args.weight = 'checkpoints/' + args.model + '_latest.pth'
     
    # prepare model
    model = prepare_model(args)

    # validate dataset & print result
    evaluator = DetectionEvaluator(args, model)
    evaluator()


if __name__ == "__main__":
    main()
