import os


import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

img = read_image('misc/14.png')
result_path = 'result/'
img = t.from_numpy(img)[None]
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('checkpoints/fasterrcnn_07291035_0.6486721503704074')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=False)
print("bbox:{}".format(_bboxes))
print("label:{}".format(_labels))
print("score:{}".format(_scores))
print(type(_bboxes))


file = open(os.path.join(result_path, '{}.txt').format(14),'w')
file.write(str(_bboxes))
file.write(str(_labels))
file.write(str(_scores))
file.close()
# vis_bbox(at.tonumpy(img[0]),
#          at.tonumpy(_bboxes[0]),
#          at.tonumpy(_labels[0]).reshape(-1),
#          at.tonumpy(_scores[0]).reshape(-1))