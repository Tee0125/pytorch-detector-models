import argparse

from helpers import DetectionTrainer
from utils import TrainerCallback

from tensorboardX import SummaryWriter


class Callback(TrainerCallback):
    def __init__(self, args):
        super().__init__()

        # print status
        print("Start Training")
        print("")

        print("Model: %s" % args.model)
        print("Batch Size: %d" % args.batch_size)
        print("Learning Rate: %.3f" % args.lr)

        if args.force_use_multiboxloss:
            print("Loss Function: Multi Box Loss")
        elif args.force_use_focalloss:
            print("Loss Function: Focal Loss")

        if args.use_step_lr:
            print("Scheduler: Step LR scheduler")
            print("Step Size: %d" % args.step_size)
            print("Gamma: %.2f" % args.gamma)
        elif args.use_multi_step_lr:
            milestones = [str(m) for m in args.milestones]

            print("Scheduler: Multi-step LR scheduler")
            print("Milestones: %s" % ", ".join(milestones))
            print("Gamma: %.2f" % args.gamma)
        elif args.use_plateau_lr:
            print("Scheduler: ReduceLROnPlateau scheduler")
        else:
            print("Scheduler: None")

        print("Momentum: %.1f" % args.momentum)
        print("")

        self.min_loss = 99.99

        self.batch_size = args.batch_size
        self.total_batches = 0

        # init tensorboard
        self.summary = SummaryWriter()

    def fit_start(self, t):
        batch_size = self.batch_size
        self.total_cnt = len(t.dataset)

    def step_start(self, t, epoch):
        print("epoch#%d start" % epoch)

    def minibatch_end(self, trainer, epoch, idx, loss):
        cnt = (idx + 1) * self.batch_size
        total_cnt = self.total_cnt

        print("[%d / %d] - loss=%.3f" % (cnt, total_cnt, loss), end='\r')

    def step_end(self, t, epoch, loss):
        print("epoch#%d end - mean loss=%.3f" % (epoch, loss))

        # write log
        self.min_loss = min(self.min_loss, loss)

        self.summary.add_scalar('loss', loss, epoch)
        self.summary.add_scalar('min_loss', self.min_loss, epoch)

        for param_group in t.optimizer.param_groups:
            self.summary.add_scalar('lr', param_group['lr'], epoch)

        # save model 
        postfix = str(epoch+1)

        if (epoch % 10) != 9:
            return

        t.save_model(postfix=postfix)

    def fit_end(self, t, loss):
        t.save_model(postfix='latest')


def main():
    parser = argparse.ArgumentParser(description='Detector Training')

    parser.add_argument('--dataset', default='VOC',
                        choices=['VOC', 'VOC2007', 'VOC2012', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default='downloads',
                        help='Dataset root directory path')
    parser.add_argument('--model', default='ssd300',
                        help='Detector model name')
    parser.add_argument('--download', default=False, action='store_true',
                        help='Download dataset')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--th_iou', default=0.5, type=float,
                        help='IOU Threshold')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--use_step_lr', default=False, action='store_true',
                        help='Use step lr scheduler')
    parser.add_argument('--step_size', default=150, type=int,
                        help='Step size for step lr scheduler')
    parser.add_argument('--use_multi_step_lr', default=False, action='store_true',
                        help='Use multi step lr scheduler')
    parser.add_argument('--milestones', default=[140, 170], type=int, nargs='*',
                        help='Milestones for multi step lr scheduler')
    parser.add_argument('--use_plateau_lr', default=False, action='store_true',
                        help='Use plateau lr scheduler')
    parser.add_argument('--force_use_multiboxloss', default=False,
                        action='store_true',
                        help='Use multi box loss')
    parser.add_argument('--force_use_focalloss', default=False,
                        action='store_true',
                        help='Use focal loss')
    parser.add_argument('--disable_augmentation', default=False, 
                        action='store_true',
                        help='Disable random augmentation')
    parser.add_argument('--enable_letterbox', default=False, 
                        action='store_true',
                        help='Enable letterboxing image')
    args = parser.parse_args()

    t = DetectionTrainer(args, callback=Callback(args))
    t.fit()


if __name__ == "__main__":
    main()

