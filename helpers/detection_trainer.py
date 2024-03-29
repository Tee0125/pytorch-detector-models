import os
import multiprocessing
import torch

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

from torch.utils.data import DataLoader

from datasets import load_dataset
from transforms import detector_transforms as transforms

from models import build_model, load_model, save_model
from losses import build_loss


class DetectionTrainer:
    def __init__(self, args, callback=None):
        self.args = args

        self.model = self.init_model()

        if args.resume:
            self.load_model(args.resume)

        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()

        self.criterion = self.init_criterion()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

        self.callback = callback

    def fit(self):
        if self.callback:
            self.callback.fit_start(self)

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(self.model).cuda()
        else:
            model = self.model

        model.train()

        for epoch in range(self.args.epochs):
            loss = self.step(model, epoch)

        if self.callback:
            self.callback.fit_end(self, loss)

    def step(self, model, epoch):
        if self.callback:
            self.callback.step_start(self, epoch)

        losses = []

        for i, batch in enumerate(self.dataloader):
            loss = self.minibatch(model, epoch, i, batch)
            losses.append(loss)

        loss = sum(losses) / len(losses)

        if self.scheduler:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        if self.callback:
            self.callback.step_end(self, epoch, loss)

        return loss
        
    def minibatch(self, model, epoch, idx, batch):
        if self.callback:
            self.callback.minibatch_start(self, epoch, idx)

        x, y = batch

        if torch.cuda.is_available():
            x = x.cuda()

        y_ = model.forward(x)
        loss = self.criterion(y, *y_)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.item()

        if self.callback:
            self.callback.minibatch_end(self, epoch, idx, loss)

        return loss

    def init_dataset(self):
        args = self.args

        size = self.model.get_input_size()

        t = []
        if not args.disable_augmentation:
            t.extend([transforms.RandomExpand(),
                      transforms.RandomSamplePatch()])
        elif args.enable_letterbox:
            t.append(transforms.LetterBox())

        t.append(transforms.Resize(size))

        if not args.disable_augmentation:
            t.extend([transforms.RandomHorizontalFlip(),
                      transforms.RandomDistort(),
                      transforms.RandomColorSpace()])

        t.extend([transforms.ToTensor(),
                  transforms.Normalize()])

        t = transforms.Compose(t)

        dataset = load_dataset(args, 
                               image_set='trainval', 
                               download=args.download,
                               transforms=t)

        return dataset

    def init_dataloader(self):
        args = self.args

        if args.num_workers < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.num_workers

        dataloader = DataLoader(self.dataset,
                                pin_memory=True,
                                shuffle=True,
                                batch_size=args.batch_size,
                                drop_last=True,
                                num_workers=num_workers,
                                collate_fn=self.collate)

        return dataloader
    
    def init_model(self):
        args = self.args

        pretrained = True if not args.resume else False

        model = build_model(args, pretrained=pretrained)

        return model

    def init_criterion(self):
        anchor = self.model.get_anchor_box()

        return build_loss(self.args, anchor=anchor)

    def init_optimizer(self):
        args = self.args

        optimizer = SGD(self.model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

        return optimizer

    def init_scheduler(self):
        if self.args.use_step_lr:
            return StepLR(self.optimizer,
                          self.args.step_size,
                          self.args.gamma)

        elif self.args.use_multi_step_lr:
            return MultiStepLR(self.optimizer,
                               self.args.milestones,
                               self.args.gamma)

        elif self.args.use_plateau_lr:
            return ReduceLROnPlateau(self.optimizer)

        elif self.args.use_cosine_lr:
            return CosineAnnealingLR(self.optimizer,
                                     self.args.epochs,
                                     1e-6)

        else:
            return None

    def load_model(self, filename):
        load_model(self.model, filename)

    def save_model(self, path='./checkpoints', postfix=None):
        save_model(self.model, path, postfix)

    @staticmethod
    def collate(batch):
        imgs = []
        targets = []

        for (img, target) in batch:
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs, 0), targets


