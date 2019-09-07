import os
import logging
import csv
 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
 
AREA_COLORE = (66, 183, 42)
 
 
class PipelineRunnerE(object):
    '''
        Very simple pipeline.
 
        Just run passed processors in order with passing context from one to 
        another.
 
        You can also set log level for processors.
    '''
 
    def __init__(self, pipelineE=None, log_level=logging.DEBUG):
        self.pipelineE = pipelineE or []
        self.contextE = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        self.set_log_level()
 
    def set_contextE(self, data):
        self.contextE = data
 
    def get_capacityE(self):
        this = self.contextE['capacityE']
        return this
 
    def add(self, processor):
        if not isinstance(processor, PipelineProcessorE):
            raise Exception(
                'Processor should be an instance of PipelineProcessorE.')
        processor.log.setLevel(self.log_level)
        self.pipelineE.append(processor)
 
    def remove(self, name):
        for i, p in enumerate(self.pipelineE):
            if p.__class__.__name__ == name:
                del self.pipelineE[i]
                return True
        return False
 
    def set_log_level(self):
        for p in self.pipelineE:
            p.log.setLevel(self.log_level)
 
    def run(self):
        for p in self.pipelineE:
            self.contextE = p(self.contextE)
 
        self.log.debug("East Frame #%d processed.", self.contextE['frame_numberE'])
 
        return self.contextE
 
 
class PipelineProcessorE(object):
    '''
        Base class for processors.
    '''
 
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        
 
class CapacityCounterE(PipelineProcessorE):
 
    def __init__(self, area_maskE, save_image=False, image_dir='./'):
        super(CapacityCounterE, self).__init__()
    
        self.area_maskE = area_maskE
        self.allE = np.count_nonzero(area_maskE)
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
        
        tE = cv2.bitwise_and(threshold,threshold,mask = self.area_maskE)
        
        freeE = np.count_nonzero(tE)
        capacityE = 1 - float(freeE)/self.allE
 
        if self.save_image:
            img = np.zeros(base_frame.shape, base_frame.dtype)
            img[:, :] = AREA_COLORE
            maskE = cv2.bitwise_and(img, img, mask=self.area_maskE)
            cv2.addWeighted(maskE, 1, base_frame, 1, 0, base_frame)
            
            fig = plt.figure()
            fig.suptitle("CapacityE: {}%".format(capacityE*100), fontsize=16)
            plt.subplot(211),plt.imshow(base_frame),plt.title('East')
            plt.xticks([]), plt.yticks([])
            plt.subplot(212),plt.imshow(tE),plt.title('East Capacity map')
            plt.xticks([]), plt.yticks([])
 
            fig.savefig(self.image_dir + ("/east_processed_%s.png" % frame_number), dpi=500)
            plt.close(fig)
            
        return capacityE
        
    def __call__(self, contextE):
        frame = contextE['frame'].copy()
        frame_numberE = contextE['frame_numberE']
        
        capacityE = self.calculate_capacity(frame, frame_numberE)
        
        self.log.debug("CapacityE: {}%".format(capacityE*100))
        contextE['capacityE'] = capacityE
        
        return contextE
