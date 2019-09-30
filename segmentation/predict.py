#from __future__ import print_function
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
import scipy

def dense_crf_v2(img, output_probs):
    w = output_probs.shape[0]
    h = output_probs.shape[1]

    output_probs = np.append(1 - output_probs, output_probs, axis=0)
    d = dcrf.DenseCRF2D(w, h, 2) #2 labels/classes
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=200, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((w,h))
        

    return Q
  

def restore_image(img=None,scale=None, size=1024.0, w_padding=0, h_padding=0, offset_x=0,offset_y=0):
    original_size= size / scale
    rescale= original_size / size
    img = img[ (h_padding +  offset_y) : (img.shape[1]-h_padding), (w_padding+offset_x) : (img.shape[0]-w_padding)]
    
    if scale < 1 :
        img = cv2.resize(img,None,fx=rescale, fy=rescale,interpolation=cv2.INTER_CUBIC)
    elif scale > 1:
        img = cv2.resize(img,None,fx=rescale, fy=rescale,interpolation=cv2.INTER_AREA)
    else:
        img = img
       
    return img


def resize_image_scale(img_path=None, min_side=1024.0, max_side=1024.0, mode='rgb'):
    if mode=='rgb':
        img = cv2.imread(img_path,1)
    else:
        img = cv2.imread(img_path,0)
    #print 'before resize:',img.shape
    if mode=='rgb':
        (rows, cols, _)=img.shape
    else:
        (rows, cols)=img.shape
    small_side = min(rows, cols)
    #print small_side
    scale = min_side / small_side
    #print scale
    large_side = max(rows, cols)
    if large_side * scale > max_side:
        scale = max_side / large_side

    if scale < 1 :
        img = cv2.resize(img,None,fx=scale, fy=scale,interpolation=cv2.INTER_AREA)
    elif scale > 1:
        img = cv2.resize(img,None,fx=scale, fy=scale,interpolation=cv2.INTER_CUBIC)
    else:
        img = img
    #print 'after resize:',img.shape
    
    
    w_padding = (max_side-img.shape[1]) / 2
    h_padding = (max_side-img.shape[0]) / 2
    w_padding=int(w_padding)
    h_padding=int(h_padding)
    
    
    if (w_padding * 2) + img.shape[1] ==max_side:
        #print 'same'
        offset_x=0
    else:
        #print 'not same'
        offset_x=max_side - (img.shape[1]+  (w_padding * 2 ))

    if (h_padding * 2) + img.shape[0] ==max_side:
        #print 'same'
        offset_y=0
    else:
        #print 'not same'
        offset_y=max_side - (img.shape[0]+  (h_padding * 2 ))  
    offset_x = int(offset_x)
    offset_y = int(offset_y)
    #print offset_x, offset_y
    if mode=='rgb':
        img = cv2.copyMakeBorder(img, h_padding+offset_y , h_padding, w_padding+offset_x, w_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img = cv2.copyMakeBorder(img, h_padding+offset_y , h_padding, w_padding+offset_x, w_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return img,scale,h_padding, w_padding, offset_x, offset_y


def custom_predict(model=None,checkpoint_dir=None,image_path=None,result_fname=None, img_size=512.0):

    bgr_img, scale, h_padding, w_padding, offset_x, offset_y=resize_image_scale(img_path=image_path,min_side=img_size, max_side=img_size)
    
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    input_data = rgb_img[None,:,:,:]

    t_start = cv2.getTickCount()
    result = model.predict(input_data, 1) #batch size 1
    Q=dense_crf_v2(bgr_img,result[0])
    
    imgMask = (result[0]*255).astype(np.uint8)
    im_mask_color = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)
    Q= Q> 0.5
    Q_mask = (Q*255).astype(np.uint8)
    im_Q_mask_color = cv2.applyColorMap(Q_mask, cv2.COLORMAP_JET)
    rescaled_Q_mask=restore_image(img=Q_mask,scale=scale,h_padding=h_padding, w_padding=w_padding, offset_x=offset_x, offset_y=offset_y,size=img_size)
    rescaled_rgb_img = restore_image(img=rgb_img,scale=scale,h_padding=h_padding, w_padding=w_padding, offset_x=offset_x, offset_y=offset_y,size=img_size)
    rescaled_org_mask=restore_image(img=imgMask,scale=scale,h_padding=h_padding, w_padding=w_padding, offset_x=offset_x, offset_y=offset_y,size=img_size)
    
    rescaled_rgb_img = cv2.cvtColor(rescaled_rgb_img, cv2.COLOR_RGB2BGR)

    return rescaled_rgb_img, rescaled_Q_mask
   