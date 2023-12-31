'''
## Reference Code Info ##
  Notebook: https://www.kaggle.com/code/rkuo2000/3dmm-fitting
  Author: RICHARD KUO
'''

## 3DMM-fitting

### Repro Github : https://github.com/Yinghao-Li/3DMM-fitting

git clone https://github.com/Yinghao-Li/3DMM-fitting
%cd 3DMM-fitting

pip install menpo menpofit menpodetect menpowidgets

## Get Images marked

### Mark frontal face image automatically.

import os
import numpy as np
import cv2
import sys

%cd test

## Start 3D fitting

sys.path.append('..')

import toml
from core import Blendshape, contour_correspondence, EdgeTopology, fitting, LandmarkMapper, Landmark, MorphableModel, utils, RenderingParameters, render

### Load and rescale pictures and landmarks

frontal_pic_name = '00029ba010_960521'
profile_pic_name = '00029pr010_940128'
frontal_img = cv2.imread(os.path.join(r'../data', frontal_pic_name + '.tif'))
profile_img = cv2.imread(os.path.join(r'../data', profile_pic_name + '.tif'))
width = np.shape(frontal_img)[1]
height = np.shape(frontal_img)[0]

s = 2000 / height if height >= width else 2000 / width
scale_param = 900 / height if height >= width else 900 / width

### Load models

morphable_model = MorphableModel.load_model(r"../py_share/py_sfm_shape_3448.bin")
blendshapes = Blendshape.load_blendshapes(r"../py_share/py_expression_blendshape_3448.bin")
landmark_mapper = LandmarkMapper.LandmarkMapper(r"../py_share/ibug_to_sfm.txt")
edge_topology = EdgeTopology.load_edge_topology(r"../py_share/py_sfm_3448_edge_topology.json")
contour_landmarks = contour_correspondence.ContourLandmarks()
contour_landmarks.load(r"../py_share/ibug_to_sfm.txt")
model_contour = contour_correspondence.ModelContour()
model_contour.laod(r"../py_share/sfm_model_contours.json")
profile_landmark_mapper = LandmarkMapper.ProfileLandmarkMapper(r"../py_share/profile_to_sfm.txt")

frontal_landmarks = []
landmark_ids = list(map(str, range(1,  69)))  # generates the numbers 1 to 68, as strings
landmarks = utils.read_pts(os.path.join(r"../data", frontal_pic_name + '.pts'))
for i in range(68):
  frontal_landmarks.append(Landmark.Landmark(landmark_ids[i], [landmarks[i][0] * s, landmarks[i][1]* s]))

proofile_landmarks = []
landmarks = utils.read_pts(os.path.join(r"../data", profile_pic_name + '.pts'))
for x in profile_landmark_mapper.right_mapper.keys():
  coor = landmarks[int(x) - 1]
  profile_landmarks.append(Landmark.Landmark(x, [coor[0]*s, [coor[1]*s]))


### Do fitting

py_mesh, frontal_rendering_params, profile_rendering_params =fitting.fit_front_and_profile(
  morphable_model, blendshape, frontal_landmarks, landmark_mapper, profile_landmarks, profile_landmark_mapper,
  round(width * s), round(height * s), edge_topology, contour_landmarks, model_contour, lambda_p=20, num_iterations=10)


### Visualize fitting result

profile_img = cv2.resize(profile_img, round(width * scale_param), round(height * scale_param)), interpolation = cv2.INTER_CUBIC)
render.draw_wireframe_with_depth(
  profile_img, py_mesh, profile_rendering_params.get_modelview(), profile_rendering_params.get_projection(),
  RenderingParameters.get_opencv_viewport(width * s, height * s), profile_landmark_mapper, scale_param / s)

frontal_img = cv2.resize(frontal_img, (round(width * scale_param), round(height * scale_param)), interpolation = cv2.INTER_CUBIC)
render.draw_wireframe_with_depth(
  frontal_img, py_mesh, frontal_rendering_params.get_modelview(), frontal_rendering_params.get_projection(),
  RenderingParameters.get_opencv_viewport(width * s, height * s), landmark_mapper, scale_param / s )

for lm in frontal_landmarks:
  cv2.rectangle(
    frontal_img, (int(lm.coordinates[0] * scale_param / s ) - 2, int(lm.coordinates[1] * scale_param / s) - 2),
    (int(lm.coordinates[0] * scale_param / s ) + 2, int(lm.coordinates[1] * scale_param /s ) + 2), (255, 0, 0))

for lm in profile_landmarks:
  cv2.rectangle(
    profile_img, (int(lm.coordiates[0] * scale_param / s ) - 2, int(lm.coordinates[1] * scale_param / s) -2 ),
    int(lm.coordinates[0]  * scale_param / s ) + 2, int(lm.coordinates[1] * scale_param / s) + 2), (255, 0, 0))


### Show fitting result
from PIL import Image
from IPython.display import display
img1 = Image.fromarray(frontal_img, 'RGB')
img2 = Image.fromarray(frontal_img, 'RGB')
display(img1, img2)


### Save result and fitted 3D model
img = np.hstack([frontal_img, profile_img])
cv2.imwrite(frontal_pic_name + '-outcome.jpg', img)
render.save_ply(py_mesh, frontal_pic_name + '-output', [210, 183, 108], author = 'Yinghao Li)
