import os
import logging
import csv
 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
 
AREA_COLORS = (66, 183, 42)
 
 
class PipelineRunnerS(object):
    '''
        Very simple pipeline.
 
        Just run passed processors in order with passing context from one to 
        another.
 
        You can also set log level for processors.
    '''
 
    def __init__(self, pipelineS=None, log_level=logging.DEBUG):
        self.pipelineS = pipelineS or []
        self.contextS = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        self.set_log_level()
 
    def set_contextS(self, data):
        self.contextS = data
 
    def get_capacityS(self):
        this = self.contextS['capacityS']
        return this
 
    def add(self, processor):
        if not isinstance(processor, PipelineProcessorS):
            raise Exception(
                'Processor should be an instance of PipelineProcessorS.')
        processor.log.setLevel(self.log_level)
        self.pipelineS.append(processor)
 
    def remove(self, name):
        for i, p in enumerate(self.pipelineN):
            if p.__class__.__name__ == name:
                del self.pipelineS[i]
                return True
        return False
 
    def set_log_level(self):
        for p in self.pipelineS:
            p.log.setLevel(self.log_level)
 
    def run(self):
        for p in self.pipelineS:
            self.contextS = p(self.contextS)
 
        self.log.debug("South Frame #%d processed.", self.contextS['frame_numberS'])
 
        return self.contextS
 
 
class PipelineProcessorS(object):
    '''
        Base class for processors.
    '''
 
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        
 
class CapacityCounterS(PipelineProcessorS):
 
    def __init__(self, area_maskS, save_image=False, image_dir='./'):
        super(CapacityCounterS, self).__init__()
    
        self.area_maskS = area_maskS
        self.allS = np.count_nonzero(area_maskS)
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
        
        tS = cv2.bitwise_and(threshold,threshold,mask = self.area_maskS)
        
        freeS = np.count_nonzero(tS)
        capacityS = 1 - float(freeS)/self.allS
 
        if self.save_image:
            img = np.zeros(base_frame.shape, base_frame.dtype)
            img[:, :] = AREA_COLORS
            maskS = cv2.bitwise_and(img, img, mask=self.area_maskS)
            cv2.addWeighted(maskS, 1, base_frame, 1, 0, base_frame)
            
            fig = plt.figure()
            fig.suptitle("CapacityS: {}%".format(capacityS*100), fontsize=16)
            plt.subplot(211),plt.imshow(base_frame),plt.title('South')
            plt.xticks([]), plt.yticks([])
            plt.subplot(212),plt.imshow(tS),plt.title('South Capacity map')
            plt.xticks([]), plt.yticks([])
 
            fig.savefig(self.image_dir + ("/south_processed_%s.png" % frame_number), dpi=500)
            plt.close(fig)
            
        return capacityS
        
    def __call__(self, contextS):
        frame = contextS['frame'].copy()
        frame_numberS = contextS['frame_numberS']
        
        capacityS = self.calculate_capacity(frame, frame_numberS)
        
        self.log.debug("CapacityS: {}%".format(capacityS*100))
        contextS['capacityS'] = capacityS
        
        return contextS
