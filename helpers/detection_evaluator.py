import os
import multiprocessing
import torch

from torch.utils.data import DataLoader

from nn import DetectPostProcess
from utils import MeanAp

from datasets import load_dataset
from transforms import detector_transforms as transforms


class DetectionEvaluator:
    def __init__(self, args, model):
        self.args = args

        self.model = model
        self.post_process = DetectPostProcess(model.get_anchor_box())

        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()

        self.data_cnt = len(self.dataset)
        self.mAP = MeanAp(len(self.dataset.classes))

    def __call__(self):
        args = self.args

        th_conf = args.th_conf
        th_iou = args.th_iou

        self.model.eval()
        self.mAP.reset()

        cnt = 0

        for i, batch in enumerate(self.dataloader):
            x, y = batch

            if torch.cuda.is_available():
                x = x.cuda()

            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                conf, loc = self.model.forward(x)
                y_ = self.post_process(conf, loc, th_iou, th_conf)

            self.match(y_, y)

            cnt += x.size(0)

        print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

        mAP, aps = self.mAP.calc_mean_ap()

        # print results
        for cls, ap in enumerate(aps):
            cls_name = self.dataset.decode_class(cls)

            print("AP(%s) = %.3f" % (cls_name, ap))

        print("mAP = %.3f" % mAP)

    def match(self, y_, y):
        for a, b in zip(y_, y):
            self.mAP.match(a, b)

    def init_dataset(self):
        args = self.args

        size = self.model.get_input_size()
        root = os.path.join(args.dataset_root, args.dataset)

        t = []
        if not args.disable_letterbox:
            t.append(transforms.LetterBox())

        t.extend([transforms.Resize(size),
                  transforms.ToTensor(),
                  transforms.Normalize()])

        t = transforms.Compose(t)

        dataset = load_dataset(args,
                               image_set='test', 
                               download=True, 
                               transforms=t)

        return dataset

    def init_dataloader(self):
        args = self.args

        if args.num_workers < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.num_workers

        return DataLoader(self.dataset,
                          pin_memory=True,
                          batch_size=args.batch_size,
                          num_workers=num_workers,
                          collate_fn=self.collate)
    
    @staticmethod
    def collate(batch):
        imgs = []
        targets = []

        for (img, target) in batch:
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs, 0), targets


