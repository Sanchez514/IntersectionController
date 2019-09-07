import logging
import logging.handlers
import os
import time
import sys
 
import cv2
import numpy as np
import skvideo.io
import utils
import matplotlib.pyplot as plt
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
 
# ============================================================================
 # Folder where the processed images go
IMAGE_DIR = "./outTest"
# Video being used for image processing
VIDEO_SOURCE = "input.mp4"
# Setting up the video into a pixel array
SHAPE = (720, 1280)
# Coordinates for North mask(Top-Right)
AREA_PTSN = np.array([[685, 135], [935, 120], [945, 135], [770, 155]])
# Coordinates for East mask(Bottom-Right
AREA_PTSE = np.array([[860, 155], [1250, 165], [1250, 190], [820, 180]])
# Coordinates for South mask(Bottom-Left)
AREA_PTSS = np.array([[425, 170], [475, 190], [200, 210], [190, 175]])
# Coordinates for West mask(Top-Left
AREA_PTSW = np.array([[210, 135], [430, 140], [370, 160], [215, 150]])
 
 
 
# Importing classes from pipelineN.py script
from pipelineN import (
    PipelineRunnerN,
    CapacityCounterN,
)
 # Importing classes from pipelineE.py script
from pipelineE import (
    PipelineRunnerE,
    CapacityCounterE,
)
 # Importing classes from pipelineS.py script
from pipelineS import (
    PipelineRunnerS,
    CapacityCounterS,
)
 # Importing classes from pipelineW.py script
from pipelineW import (
    PipelineRunnerW,
    CapacityCounterW,
)
 # Importing classes from lightcontrol.py script
from lightcontrol import (
    SetLights,
    KoopsTurn,
    TurnOff,
)
# ============================================================================
 
 
def main():
   # Set up lightcontrol.py variables
    setit = SetLights()
    change = KoopsTurn()
    off = TurnOff()
 
    # Set up the lights
    setit.run()
    
    log = logging.getLogger("main")
 
    # Setup all 4 lanes area masks
    baseN = np.zeros(SHAPE + (3,), dtype='uint8')
    baseE = np.zeros(SHAPE + (3,), dtype='uint8')
    baseS = np.zeros(SHAPE + (3,), dtype='uint8')
    baseW = np.zeros(SHAPE + (3,), dtype='uint8')
 
    area_maskN = cv2.fillPoly(baseN, [AREA_PTSN], (255, 255, 255))[:, :, 0]
    area_maskE = cv2.fillPoly(baseE, [AREA_PTSE], (255, 255, 255))[:, :, 0]
    area_maskS = cv2.fillPoly(baseS, [AREA_PTSS], (255, 255, 255))[:, :, 0]
    area_maskW = cv2.fillPoly(baseW, [AREA_PTSW], (255, 255, 255))[:, :, 0]
 
    #Collect variables from the imported .py scripts
    pipelineN = PipelineRunnerN(pipelineN=[
        CapacityCounterN(area_maskN=area_maskN, save_image=True, image_dir=IMAGE_DIR),
        # saving every 10 seconds
    ], log_level=logging.DEBUG)
 
    pipelineE = PipelineRunnerE(pipelineE=[
        CapacityCounterE(area_maskE=area_maskE, save_image=True, image_dir=IMAGE_DIR),
        # saving every 10 seconds
    ], log_level=logging.DEBUG)
 
    pipelineS = PipelineRunnerS(pipelineS=[
        CapacityCounterS(area_maskS=area_maskS, save_image=True, image_dir=IMAGE_DIR),
        # saving every 10 seconds
    ], log_level=logging.DEBUG)
 
    pipelineW = PipelineRunnerW(pipelineW=[
        CapacityCounterW(area_maskW=area_maskW, save_image=True, image_dir=IMAGE_DIR),
        # saving every 10 seconds
    ], log_level=logging.DEBUG)
 
 
 
    # Set up image source
    cap = skvideo.io.vreader(VIDEO_SOURCE)
    
    # setup the frames so that it goes to the next one for all 4 lanes
    frame_numberN = -1
    st = time.time()
    frame_numberE = -1
    frame_numberS = -1
    frame_numberW = -1
    
    # Loop getting the capacities
    try:
        for frame in cap:
            if not frame.any():
                log.error("Frame capture failed, skipping...")
# increment the frames to cycle through them all
            frame_numberN += 1
            frame_numberE += 1
            frame_numberS += 1
            frame_numberW += 1
 
 # run the 4 pipeline python scripts
            pipelineN.set_contextN({
                'frame': frame,
                'frame_numberN': frame_numberN,
            })
            contextN = pipelineN.run()
 
            pipelineE.set_contextE({
                'frame': frame,
                'frame_numberE': frame_numberE,
            })
            
            contextE = pipelineE.run()
 
            pipelineS.set_contextS({
                'frame': frame,
                'frame_numberS': frame_numberS,
            })
            
            contextS = pipelineS.run()
 
            pipelineW.set_contextW({
                'frame': frame,
                'frame_numberW': frame_numberW,
            })
            
            contextW = pipelineW.run()
 
 # Get capacity variable from the pipeline scripts
            capacityN = pipelineN.get_capacityN()
            capacityE = pipelineE.get_capacityE()
            capacityS = pipelineS.get_capacityS()
            capacityW = pipelineW.get_capacityW()
 
# Change LEDs based off of capacities
            if(capacityE>capacityN):
                change.run()
 
            # skipping 10 seconds
            for i in range(240):
                next(cap)
 
    except Exception as e:
        log.exception(e)
            
    
 
# ============================================================================
 
if __name__ == "__main__":
    log = utils.init_logging()
 
    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)
        off.run()
    main()
