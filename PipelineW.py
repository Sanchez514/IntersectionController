import os
import logging
import csv
 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
 
AREA_COLORW = (66, 183, 42)
 
 
class PipelineRunnerW(object):
    '''
        Very simple pipeline.
 
        Just run passed processors in order with passing context from one to 
        another.
 
        You can also set log level for processors.
    '''
 
    def __init__(self, pipelineW=None, log_level=logging.DEBUG):
        self.pipelineW = pipelineW or []
        self.contextW = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        self.set_log_level()
 
    def set_contextW(self, data):
        self.contextW = data
 
    def get_capacityW(self):
        this = self.contextW['capacityW']
        return this
 
    def add(self, processor):
        if not isinstance(processor, PipelineProcessorW):
            raise Exception(
                'Processor should be an instance of PipelineProcessorW.')
        processor.log.setLevel(self.log_level)
        self.pipelineW.append(processor)
 
    def remove(self, name):
        for i, p in enumerate(self.pipelineW):
            if p.__class__.__name__ == name:
                del self.pipelineW[i]
                return True
        return False
 
    def set_log_level(self):
        for p in self.pipelineW:
            p.log.setLevel(self.log_level)
 
    def run(self):
        for p in self.pipelineW:
            self.contextW = p(self.contextW)
 
        self.log.debug("West Frame #%d processed.", self.contextW['frame_numberW'])
 
        return self.contextW
 
 
class PipelineProcessorW(object):
    '''
        Base class for processors.
    '''
 
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        
 
class CapacityCounterW(PipelineProcessorW):
 
    def __init__(self, area_maskW, save_image=False, image_dir='./'):
        super(CapacityCounterW, self).__init__()
    
        self.area_maskW = area_maskW
        self.allW = np.count_nonzero(area_maskW)
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
        
        tW = cv2.bitwise_and(threshold,threshold,mask = self.area_maskW)
        
        freeW = np.count_nonzero(tW)
        capacityW = 1 - float(freeW)/self.allW
 
        if self.save_image:
            img = np.zeros(base_frame.shape, base_frame.dtype)
            img[:, :] = AREA_COLORW
            maskW = cv2.bitwise_and(img, img, mask=self.area_maskW)
            cv2.addWeighted(maskW, 1, base_frame, 1, 0, base_frame)
            
            fig = plt.figure()
            fig.suptitle("CapacityW: {}%".format(capacityW*100), fontsize=16)
            plt.subplot(211),plt.imshow(base_frame),plt.title('West')
            plt.xticks([]), plt.yticks([])
            plt.subplot(212),plt.imshow(tW),plt.title('West Capacity map')
            plt.xticks([]), plt.yticks([])
 
            fig.savefig(self.image_dir + ("/west_processed_%s.png" % frame_number), dpi=500)
            plt.close(fig)
            
        return capacityW
        
    def __call__(self, contextW):
        frame = contextW['frame'].copy()
        frame_numberW = contextW['frame_numberW']
        
        capacityW = self.calculate_capacity(frame, frame_numberW)
        
        self.log.debug("CapacityW: {}%".format(capacityW*100))
        contextW['capacityW'] = capacityW
        
        return contextW
