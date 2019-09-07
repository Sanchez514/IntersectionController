import os
import logging
import csv
 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
 
AREA_COLORN = (66, 183, 42)
 
 
class PipelineRunnerN(object):
    '''
        Very simple pipeline.
 
        Just run passed processors in order with passing context from one to 
        another.
 
        You can also set log level for processors.
    '''
 
    def __init__(self, pipelineN=None, log_level=logging.DEBUG):
        self.pipelineN = pipelineN or []
        self.contextN = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        self.set_log_level()
 
    def set_contextN(self, data):
        self.contextN = data
 
    def get_capacityN(self):
        this = self.contextN['capacityN']
        return this
 
    def add(self, processor):
        if not isinstance(processor, PipelineProcessorN):
            raise Exception(
                'Processor should be an instance of PipelineProcessorN.')
        processor.log.setLevel(self.log_level)
        self.pipelineN.append(processor)
 
    def remove(self, name):
        for i, p in enumerate(self.pipelineN):
            if p.__class__.__name__ == name:
                del self.pipelineN[i]
                return True
        return False
 
    def set_log_level(self):
        for p in self.pipelineN:
            p.log.setLevel(self.log_level)
 
    def run(self):
        for p in self.pipelineN:
            self.contextN = p(self.contextN)
 
        self.log.debug("North Frame #%d processed.", self.contextN['frame_numberN'])
 
        return self.contextN
 
 
class PipelineProcessorN(object):
    '''
        Base class for processors.
    '''
 
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        
 
class CapacityCounterN(PipelineProcessorN):
 
    def __init__(self, area_maskN, save_image=False, image_dir='./'):
        super(CapacityCounterN, self).__init__()
    
        self.area_maskN = area_maskN
        self.allN = np.count_nonzero(area_maskN)
        self.image_dir = image_dir
        self.save_image = save_image
    
        
    def calculate_capacity(self, frame, frame_number):
        base_frame = frame
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # this used for noise reduction at night time
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(frame)
    
        edges = cv2.Canny(frame,50,70)
        edges = ~edges
        blur = cv2.bilateralFilter(cv2.blur(edges,(21,21), 100),9,200,200)
        _, threshold = cv2.threshold(blur,230, 255,cv2.THRESH_BINARY)
        
        tN = cv2.bitwise_and(threshold,threshold,mask = self.area_maskN)
        
        freeN = np.count_nonzero(tN)
        capacityN = 1 - float(freeN)/self.allN
 
        if self.save_image:
            img = np.zeros(base_frame.shape, base_frame.dtype)
            img[:, :] = AREA_COLORN
            maskN = cv2.bitwise_and(img, img, mask=self.area_maskN)
            cv2.addWeighted(maskN, 1, base_frame, 1, 0, base_frame)
            
            fig = plt.figure()
            fig.suptitle("CapacityN: {}%".format(capacityN*100), fontsize=16)
            plt.subplot(211),plt.imshow(base_frame),plt.title('North')
            plt.xticks([]), plt.yticks([])
            plt.subplot(212),plt.imshow(tN),plt.title('North Capacity map')
            plt.xticks([]), plt.yticks([])
 
            fig.savefig(self.image_dir + ("/north_processed_%s.png" % frame_number), dpi=500)
            plt.close(fig)
            
        return capacityN
        
    def __call__(self, contextN):
        frame = contextN['frame'].copy()
        frame_numberN = contextN['frame_numberN']
        
        capacityN = self.calculate_capacity(frame, frame_numberN)
        
        self.log.debug("CapacityN: {}%".format(capacityN*100))
        contextN['capacityN'] = capacityN
        
        return contextN
