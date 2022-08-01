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


result_path = 'result/'
for img in imgs:
    _id = img.split('/')[-1].split('.')[0]
    img = read_image(img)
    id_image = t.from_numpy(img)[None]
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('checkpoints/fasterrcnn_07291035_0.6486721503704074')
    opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(id_image, visualize=True)

    # print('_bboxes:{}'.format(_bboxes))
    # print('_labels:{}'.format(_labels))
    # print('_scores:{}'.format(_scores))
    file = open(os.path.join(result_path, '{}.txt').format(_id), 'w')
    for i in range(0, len(_labels[0])):
        file.write(str(_labels[0][i]) + ' ')
        file.write(str(_bboxes[0][i]) + ' ')
        file.write(str(_scores[0][i]) + ' ')
    file.close()





