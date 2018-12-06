import argparse
import sys
import time

from models import *
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
parser.add_argument('-batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('-data_config_path', type=str, default='cfg/coco.data', help='data config file path')
parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-multi_scale', default=False, help='random image sizes per batch 320 - 608')
parser.add_argument('-img_size', type=int, default=32 * 13, help='pixels')
parser.add_argument('-resume', default=False, help='resume training flag')
parser.add_argument('-batch_report', default=False, help='report TP, FP, FN, P and R per batch (slower)')
parser.add_argument('-freeze_darknet53', default=False, help='freeze darknet53.conv.74 layers for first epoch')
parser.add_argument('-var', type=float, default=0, help='optional test variable')
opt = parser.parse_args()
if opt.multi_scale:  # pass maximum multi_scale size
    opt.img_size = 608
print(opt)

# Import test.py to get mAP after each epoch
sys.argv[1:] = []  # delete any train.py command-line arguments before they reach test.py
import test  # must follow sys.argv[1:] = []

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if not opt.multi_scale:
        torch.backends.cudnn.benchmark = True


def main(opt):
    os.makedirs('weights', exist_ok=True)

    # Configure run
    data_config = parse_data_config(opt.data_config_path)
    num_classes = int(data_config['classes'])
    train_path = '../coco/trainvalno5k.txt'

    # Initialize model
    model = Darknet(opt.cfg, opt.img_size)

    # Get dataloader
    dataloader = load_images_and_labels(train_path, batch_size=opt.batch_size, img_size=opt.img_size,
                                        multi_scale=opt.multi_scale, augment=True)

    lr0 = 0.001
    if opt.resume:
        checkpoint = torch.load('weights/latest.pt', map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        if torch.cuda.device_count() > 1:
            raise Exception('Multi-GPU not currently supported: https://github.com/ultralytics/yolov3/issues/21')
            # print('Using ', torch.cuda.device_count(), ' GPUs')
            # model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        start_epoch = 0
        best_loss = float('inf')

        # Initialize model with darknet53 weights (optional)
        if not os.path.isfile('weights/darknet53.conv.74'):
            os.system('wget https://pjreddie.com/media/files/darknet53.conv.74 -P weights')
        load_weights(model, 'weights/darknet53.conv.74')

        if torch.cuda.device_count() > 1:
            raise Exception('Multi-GPU not currently supported: https://github.com/ultralytics/yolov3/issues/21')
            # print('Using ', torch.cuda.device_count(), ' GPUs')
            # model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    model_info(model)
    t0, t1 = time.time(), time.time()
    mean_recall, mean_precision = 0, 0
    print('%11s' * 16 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nTargets', 'TP', 'FP', 'FN', 'time'))
    for epoch in range(opt.epochs):
        epoch += start_epoch

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        if epoch > 50:
            lr = lr0 / 10
        else:
            lr = lr0
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Freeze darknet53.conv.74 layers for first epoch
        if opt.freeze_darknet53:
            if epoch == 0:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif epoch == 1:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = True

        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(3, num_classes)
        optimizer.zero_grad()
        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                lr = lr0 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, batch_report=opt.batch_report, var=opt.var)
            loss.backward()

            # accumulated_batches = 1  # accumulate gradient for 4 batches before stepping optimizer
            # if ((i+1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
            optimizer.step()
            optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            if opt.batch_report:
                TP, FP, FN = metrics
                metrics += model.losses['metrics']

                # Precision
                precision = TP / (TP + FP)
                k = (TP + FP) > 0
                if k.sum() > 0:
                    mean_precision = precision[k].mean()

                # Recall
                recall = TP / (TP + FN)
                k = (TP + FN) > 0
                if k.sum() > 0:
                    mean_recall = recall[k].mean()

            s = ('%11s%11s' + '%11.3g' * 14) % (
                '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                rloss['loss'], mean_precision, mean_recall, model.losses['nT'], model.losses['TP'],
                model.losses['FP'], model.losses['FN'], time.time() - t1)
            t1 = time.time()
            print(s)

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'weights/latest.pt')

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp weights/latest.pt weights/best.pt')

        # Save backup weights every 5 epochs
        if (epoch > 0) & (epoch % 5 == 0):
            os.system('cp weights/latest.pt weights/backup' + str(epoch) + '.pt')

        # Calculate mAP
        test.opt.weights_path = 'weights/latest.pt'
        mAP, R, P = test.main(test.opt)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
