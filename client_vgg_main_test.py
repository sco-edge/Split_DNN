
import sys
import argparse
import subprocess
import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
import pickle
import json

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
#from keyFrameDetection import KeyFrameDetection
from communication import clientCommunication
from muLinUCB import muLinUCB
from yolo_utils import load_class_names, get_boxes, plot_boxes_cv2

import torch.nn as nn
import torch

vgg_info = { # action No. : [layer type num{1: conv, 2: fc, 3: act}, total mac{1: conv, 2: fc, 3: act}, mid_data_size, partition point]
                0: [13, 3, 24, 15346630656, 123633664, 26208256, 4818272, 0],
                1: [12, 3, 23, 15259926528, 123633664, 22996992, 102761824, 1],
                2: [11, 3, 22, 13410238464, 123633664, 19785728, 102761824, 2],
                3: [11, 3, 21, 13410238464, 123633664, 16574464, 25691488, 3],
                4: [10, 3, 20, 12485394432, 123633664, 13363200, 51381600, 4],
                5: [9, 3, 19, 10635706368, 123633664, 10151936, 51381600, 5],
                6: [9, 3, 18, 10635706368, 123633664, 8546304, 12846432, 6],
                7: [8, 3, 17, 9710862336, 123633664, 6940672, 25691496, 7],
                8: [7, 3, 16, 7861174272, 123633664, 5335040, 25691496, 8],
                9: [6, 3, 15, 6011486208, 123633664, 4532224, 25691496, 9],
                10: [6, 3, 14, 6011486208, 123633664, 3729408, 6423912, 10],
                11: [5, 3, 13, 5086642176, 123633664, 2926592, 12846440, 11],
                12: [4, 3, 12, 3236954112, 123633664, 2123776, 12846440, 12],
                13: [3, 3, 11, 1387266048, 123633664, 1320960, 12846440, 13],
                14: [3, 3, 10, 1387266048, 123633664, 919552, 3212648, 14],
                15: [2, 3, 9, 924844032, 123633664, 518144, 3212648, 15],
                16: [1, 3, 8, 462422016, 123633664, 417792, 3212648, 16],
                17: [0, 3, 7, 0, 123633664, 317440, 3212648, 17],
                18: [0, 3, 6, 0, 123633664, 217088, 3212648, 18],
                19: [0, 3, 4, 0, 123633664, 16384, 804200, 19],
                20: [0, 2, 2, 0, 20873216, 12288, 804200, 20],
                21: [0, 1, 0, 0, 4096000, 0, 132416, 21],
                22: [0, 0, 0, 0, 0, 0, 0, 22]
                }



def getActualDelay(action, model, preprocessed_image, totallayerNo, communication):
    if action == totallayerNo - 1: # local mobile process
        start_time1 = time.time()
        prediction = model(preprocessed_image)
        end_time1 = time.time()
        return end_time1 - start_time1, 0, prediction
    else:
        start_time1 = time.time()
        intermediate_output = model(preprocessed_image, server=False, partition=action)
        end_time1 = time.time()

    

    result = intermediate_output.data 
    #data_to_server = [action, intermediate_output.data]
    del intermediate_output

    start_time = time.time()
    #communication.send_msg(data_to_server)

    #result = communication.receive_msg()

    #communication.close_channel()
    end_time = time.time()

    return end_time1 - start_time1, end_time - start_time,  result



if __name__ == '__main__':
    print('test partition points in vgg16!!!')

    import json
    import torchvision.transforms as transforms
    from PIL import Image

    with open("imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        labels = {int(key): value for key, value in class_idx.items()}

    model = vgg16()
    model.eval()
    partitionInfo = vgg_info
    Action_num = len(partitionInfo)
    if torch.cuda.is_available():
        model.cuda()

    min_img_size = 224
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    img = Image.open('Golden_Retriever_Hund_Dog.jpg')
    img = transform_pipeline(img)
    img = img.unsqueeze(0)

    #communication = clientCommunication(args.host, args.port)
    communication = clientCommunication('127.0.0.1', 8080)

    for partition in range(2):
        with torch.no_grad():
            
            #intermediate = model(img.cuda(), server=False, partition=partition)
            #front_end_delay, actual_comm_delay, res = getActualDelay(partition, model, img, Action_num, communication)

            intermediate = model(img, server=False, partition=partition)
            #prediction = model(intermediate, server=True, partition=partition)

            #prediction = torch.argmax(prediction)
            prediction = torch.argmax(intermediate)
            #prediction = torch.argmax(res)

            print('partition point ', partition, labels[prediction.item()])
            #print('partition point ', partition, labels[prediction.item()], front_end_delay, actual_comm_delay)
            #print(partition, ',', front_end_delay, ',', actual_comm_delay)
