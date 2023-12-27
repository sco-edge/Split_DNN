import argparse
import cv2
import torchvision.transforms as transforms
import torch
import time

from PIL import Image

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from communication import serverCommunication


WINDOW_NAME = 'CameraDemo'


def parse_args():
    # Parse input arguments
    desc = 'ANS in edge server side'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dnn', dest='dnn_model',
                        help='vgg, yolo',
                        default='vgg', type=str)
    parser.add_argument('--host', dest='host',
                        help='Ip address',
                        default='127.0.0.1', type=str)
    parser.add_argument('--port', dest='port',
                        help='Ip port',
                        default=8080, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
    else:
        model = tinyYolo()
        model.eval()

    #model.cuda()

    communication = serverCommunication(args.host, args.port)

    while True:
        try:
            conn, addr = communication.accept_conn()
            with conn:
                recv_data = communication.receive_msg(conn)
                #print('receive data from mobile device !!!')
                partition_point = recv_data[0]
                data = recv_data[1]
                data = torch.autograd.Variable(data)
                #prediction = model(data.cuda(), server=True, partition=partition_point)
                start_time2 = time.time()
                prediction = model(data, server=True, partition=partition_point)
                end_time2= time.time()
                #print('partition point', partition_point, end_time2-start_time2)
                print(partition_point, ',', end_time2-start_time2)
                res = prediction.data

                msg = communication.send_msg(conn, res)

        except KeyboardInterrupt or TypeError or OSError:
            communication.close_channel()
