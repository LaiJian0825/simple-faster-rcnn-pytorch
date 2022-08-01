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

# image_path = '/dev/validation/'
image_path = '/media/dataset/test/ali_test_nanning/'
imgs = glob.glob(os.path.join(image_path,'*.png'))
print(imgs)
result_path = 'result/'
for img in imgs:
    id = img.split('/')[-1]
    print(id)
    img = read_image(img, color=True)
    print(img)
    id_image = t.from_numpy(img)[None]
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('checkpoints/fasterrcnn_07291235_0.689377591224362')
    opt.caffe_pretrain = True  # this model was trained from torchvision-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    # id_list_file = os.path.join(
    #     data_dir, 'ImageSets/Main/{0}.txt'.format(split))
    # file = open(result_path, 'w')
    file = open(os.path.join(result_path, '{}.txt').format(id))
    file.write(str(_labels[0][0]) + ' ')
    file.write(str(_bboxes[0][0]) + ' ')
    file.write(str(_scores[0][0]) + ' ')
    file.close()


