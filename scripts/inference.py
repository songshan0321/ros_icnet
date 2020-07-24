import tensorflow as tf
import numpy as np
import cv2
import time

from model import ICNet_BN

class Config(object):

    def __init__(self, is_training=False, filter_scale=1):

        self.param = {'name': 'lbfreespace',
                    'num_classes': 2,
                    'ignore_label': 255,
                    'infer_size': [720, 960],
                    'eval_size': [720, 960],
                    'eval_steps': 114}

        # self.num_classes = 2
        self.infer_size = (720, 960, 3)
        self.is_training = is_training
        self.filter_scale = filter_scale

class ICNetInference:
    
    def __init__(self, weight_path):
        self.cfg = Config()
        self.cfg.model_weight = weight_path
        self.net = ICNet_BN(cfg=self.cfg, mode='inference')
        self.net.create_session()
        self.net.restore(self.cfg.model_weight)

    def infer(self, image):
        dim = (self.cfg.param['infer_size'][1], self.cfg.param['infer_size'][0])
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return self.net.predict(image)

    def process(self, image, result, alpha=0.3):
        dim = (self.cfg.param['infer_size'][1], self.cfg.param['infer_size'][0])
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
        overlay = (0.5 * image + 0.5 * result).astype("uint8")

        result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)/255.0
        result = (abs(result-1))
        result = np.uint8(result)
        output, boundary = self.extract_boundary_np(overlay, result)

        return output, boundary

    def extract_boundary_np(self, image, result):
        result[0:int(0.45*result.shape[0]),:] = 1
        mask = cv2.copyMakeBorder(result, 0, 1, 0, 0, cv2.BORDER_CONSTANT, None, [0,0,0])
        edge = mask[:-1,:] - mask[1:,:]
        points = np.nonzero(edge)
        points = np.transpose(points)
        idx = np.lexsort((-points[:,0],points[:,1]))  
        sorted_idx = points[idx]
        
        unique_idx = np.unique(sorted_idx[:,1],return_index=True,axis=0)[1]
        unique_idx = sorted_idx[unique_idx]
        
        for p in unique_idx:
            image_draw = cv2.circle(image, (p[1],p[0]), 1, (255, 255, 0) , 5)

        return image_draw, unique_idx