import argparse
import time
import os
import numpy as np
from tqdm import tqdm

from models import *
from utils.datasets import *
from utils.utils import *

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
f_path = os.path.dirname(os.path.realpath(__file__)) + '/'

parser = argparse.ArgumentParser()
# Get data configuration

parser.add_argument('-image_folder', type=str, default='data/samples', help='path to images')
# parser.add_argument('-image_folder', type=str, default='./data/samples/37.jpeg', help='path to images')
# parser.add_argument('-image_folder', type=str, default='./data/samples/zidane.jpg', help='path to images')
parser.add_argument('-output_folder', type=str, default='feats_out', help='path to outputs')
parser.add_argument('-plot_flag', type=bool, default=True)
parser.add_argument('-txt_out', type=bool, default=False)

parser.add_argument('-cfg', type=str, default=f_path + 'cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-class_path', type=str, default=f_path + 'data/coco.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.50, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=32 * 13, help='size of each image dimension')
opt = parser.parse_args()
print(opt)


def main(opt):
    os.system('rm -rf ' + opt.output_folder)
    os.makedirs(opt.output_folder, exist_ok=True)

    # Load model
    model = Darknet(opt.cfg, opt.img_size)

    weights_path = f_path + 'weights/yolov3.pt'
    #weights_path = f_path + 'weights/yolov3.weights'
    # weights_path = f_path + 'weights/darknet53.conv.74'
    if weights_path.endswith('.pt'):  # pytorch format
        if weights_path.endswith('weights/yolov3.pt') and not os.path.isfile(weights_path):
            os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights_path)
        # checkpoint = torch.load(weights_path, map_location='cpu')
        # checkpoint = torch.load(weights_path, map_location='cuda:0')
        # model.load_state_dict(checkpoint['model'])
        # del checkpoint
    else:  # darknet format
        load_weights(model, weights_path)

        checkpoint = torch.load(weights_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model'])
        del checkpoint

    model.to(device).eval()

    files_root='../aim_sets/' #158 in total
    prev_time = time.time()
    aim_uris=sorted(os.listdir(files_root))
    print('total uris:',aim_uris)
    for uri in aim_uris:
        views=os.listdir(files_root+uri+'/')
        for view in views: #different view of one meeting
            cutted_samples=[name.replace('.wav','').replace('.avi','').replace('.npy','') for name in os.listdir(files_root+uri+'/'+view)]
            ids=sorted(set(cutted_samples))
            assert len(cutted_samples)==3*len(ids)
            for id in ids: # different part of the meeting
                try:
                    image_folder=files_root+uri+'/'+view+'/'+id
                    print('*******',image_folder)

                    # Set Dataloader
                    dataloader = load_images(image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

                    data_holder=[]
                    for i, (img_paths, img) in enumerate(dataloader):
                        # print('%g/%g' % (i + 1, len(dataloader)), end=' ')
                        # print(img_paths)

                        with torch.no_grad():
                            pred,features = model(torch.from_numpy(img).unsqueeze(0).to(device))
                            # print(features.size())
                            features=features.data.cpu().numpy()
                            data_holder.append(features)
                        # print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
                    data_holder=np.array(data_holder)
                    np.save(image_folder+'.npy',data_holder)
                except Exception as rr:
                    print('EEE',rr)
                    continue

    print('Takes total:',time.time()-prev_time)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
