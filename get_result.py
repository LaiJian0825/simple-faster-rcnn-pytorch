import glob
import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

# import os
# import glob
# WSI_MASK_PATH = 'E:\\contest\\insect\\insect\\JPEGImages\\test'#存放图片的文件夹路径
# paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpeg'))
# print(paths)
# paths.sort()
# print(paths)

image_path = r'/media/dataset/VOCcad/JPEGImages/'
imgs = glob.glob(os.path.join(image_path,'*.png'))
result_path = r'/root/projects/simple-faster-rcnn-pytorch/result/'
for img in imgs:
    id = img.split('/')[-1]
    id_image = t.from_numpy(img)[None]
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('checkpoints/fasterrcnn_07291035_0.6486721503704074')
    opt.caffe_pretrain = False # this model was trained from torchvision-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    # id_list_file = os.path.join(
    #     data_dir, 'ImageSets/Main/{0}.txt'.format(split))
    # file = open(result_path, 'w')
    file = open(os.path.join(result_path, '{}.txt').format(id))
    file.write(_bboxes)
    file.write(_labels)
    file.write(_scores)


