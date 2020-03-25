from .voc_detection import VOCDetection


def load_dataset(args, image_set='train', download=True, transforms=None):
    dataset = args.dataset
    root = args.dataset_root

    if dataset[0:3] == 'VOC':
        if dataset == 'VOC2007':
            year = ('2007',)
        elif dataset == 'VOC2012':
            year = ('2012',)
        elif image_set == 'test':
            year = ('2007',)
        else:
            year = ('2007', '2012')

        return VOCDetection(root,
                            year=year,
                            image_set=image_set,
                            download=download,
                            transforms=transforms)

    raise Exception("Unknown dataset %s" % dataset)

