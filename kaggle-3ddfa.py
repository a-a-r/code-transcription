'''
# 3D Dense Face Alignment/Reconstruction w. 3DDFA_V2 #
Author : BALRAJ ASHWATH
https://www.kaggle.com/code/balraj98/3d-dense-face-alignment-reconstruction-w-3ddfa-v2
'''

### Introduction

# The notebook is a demo of Towrards Fast, Accurate and Stable 3D Dense Face Alignment obtained from the authors' original 3DDFA_V2 Implementation.

### Acknoweldgements

# This work was inspired by and borrows code from Jianzhu Guo's 3DDFA_V2 Implementation. 
# If you use this work, you should cite the research work Towards Fast, Accurate and Stable 3D Dense Face Alignment and cite / star the official implementation.

!git clone https://github.com/cleardusk/3DDFA_V2.git
%cd 3DDFA_V2
!sh ./build.sh

# before import, mask sure FaceBoxes and Sim3DR are built successfully

import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix

import matplotlib.pyplot as plt
from skimage import io
from IPython.display import Image


### Load configs

# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True    # or True to use ONNX to speed up
if onnx_flag:
    !pip install onnxruntime

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)

else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode = False, **cfg)


# given an image path or the image url

# img_fp = 'examples/inputs/emma.jpg'
# img = cv2.imread(img_fp)
# plt.imshow(img[..., ::-1])

img_url = 'https://raw.githubusercontent.com/cleardusk/3DDFA_V2/master/examples/inputs/emma.jpg'
img = io.imread(img_url)
plt.imshow(img)
plt.axis('off')

img = img[..., ::-1]    # RGB -> BGR


### Detect faces using FaceBoxes

# face detection
boxes = face_boxes(img)
print(f'Detect {len(boxes)} faces')
print(boxes)


### Regressing 3DMM parameters, reconstruction and visualization

# regress 3DMM params
param_lst, roi_box_lst = tddfa(img, boxes)

# reconstruct vertices and visualizing sparse landmarks
dense_flag = False
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
draw_landmarks(img, ver_lst, dense_flag=dense_flag)

# reconstruct vertices and visualizing dense landmarks
dense_flag = True
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
draw_landmarks(img, ver_lst, dense_flag=dense_flag)

# reconstruct vertices and render
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=True);

# reconstruct vertices and render depth
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
depth(img, ver_lst, tddfa.tri, show_flag=True);

# reconstruct vertices and render pncc
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
pncc(img, ver_lst, tddfa.tri, show_flag=True);

# running offline
# %%bash
for OPT in ['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'pose', 'uv_tex', 'ply', 'obj']:
    !python demo.py -f examples/inputs/trump_hillary.jpg -o $OPT --show_flag=false --onnx;


!ls examples/results/

''' [Output]
trump_hillary_2d_dense.jpg   trump_hillary_obj.obj   trump_hillary_uv_tex.jpg
trump_hillary_2d_sparse.jpg  trump_hillary_ply.ply   videos
trump_hillary_3d.jpg	     trump_hillary_pncc.jpg
trump_hillary_depth.jpg      trump_hillary_pose.jpg
'''

Image('examples/results/trump_hillary_2d_dense.jpg')
Image('examples/results/trump_hillary_2d_sparse.jpg')
Image('examples/results/trump_hillary_3d.jpg')
Image('examples/results/trump_hillary_depth.jpg')
Image('examples/results/trump_hillary_pncc.jpg')
Image('examples/results/trump_hillary_pose.jpg')

!rm -r examples/
