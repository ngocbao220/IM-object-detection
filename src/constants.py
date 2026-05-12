VOC_CLASSES = [
    "__background__", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES)
