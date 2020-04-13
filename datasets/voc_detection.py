from torchvision.datasets import VOCDetection as VOCDetection_


class VOCDetection:
    classes = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        )

    def __init__(self,
                 root,
                 year='2007',
                 image_set='train',
                 download=False,
                 transforms=None,
                 ignore_difficult=True):

        self.datasets = []
        self.cnt_per_dataset = []

        self.ignore_difficult = ignore_difficult

        if not isinstance(year, list) and not isinstance(year, tuple):
            year = ( year, )

        for _year in year:
            d = VOCDetection_(root,
                              year=_year,
                              image_set=image_set,
                              download=download)

            self.datasets.append(d)
            self.cnt_per_dataset.append(len(d))

        self.class2idx = {}
        for idx, name in enumerate(self.classes):
            self.class2idx[name] = idx

        self.transforms = transforms

    def __getitem__(self, index):
        for dataset, cnt in zip(self.datasets, self.cnt_per_dataset):
            if index < cnt:
                img, target = dataset[index]
                break

            index -= cnt

        size = target['annotation']['size']

        w = float(size['width'])
        h = float(size['height'])

        objs = target['annotation']['object']
        if isinstance(objs, dict):
            objs = [objs]

        target = []
        for obj in objs:
            if self.ignore_difficult and int(obj['difficult']):
                continue

            label = self.encode_class(obj['name'])

            x1 = float(obj['bndbox']['xmin']) / w
            y1 = float(obj['bndbox']['ymin']) / h
            x2 = float(obj['bndbox']['xmax']) / w
            y2 = float(obj['bndbox']['ymax']) / h

            target.append([x1, y1, x2, y2, label])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return sum(self.cnt_per_dataset)

    def encode_class(self, class_name):
        return self.class2idx[class_name]

    def decode_class(self, class_id):
        return self.classes[class_id]
