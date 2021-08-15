import argparse
import numpy as np
from tqdm import tqdm
import torch
from dataloader import data_loader
from models.unet import UNet
from models.unet_equiconv import UNetEquiconv
from utils.metrics import Evaluator


class Test(object):
    """UNet-stdconv and UNet-equiconv test"""
    def __init__(self, args):
        self.args = args
        self.test_loader = data_loader(args)
        self.model = None

        if args.conv_type == 'Std':
            print("UNet-stdconv")
            self.model = UNet(args.num_classes)
            state_dict = torch.load("weights/stdconv.pth.tar")
        elif args.conv_type == "Equi":
            print("UNet-equiconv")
            layerdict, offsetdict = torch.load('layer_256x512.pt'), torch.load('offset_256x512.pt')
            self.model = UNetEquiconv(args.num_classes, layer_dict=layerdict, offset_dict=offsetdict)
            state_dict = torch.load("weights/equiconv.pth.tar")
        else:
            raise Exception

        self.model.load_state_dict(state_dict)
        self.evaluator = Evaluator(args.num_classes)
        if args.cuda:
            self.model = self.model.cuda()

    def run(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        with torch.no_grad():
            for _, sample in enumerate(tbar):
                image, target = sample['image'], sample['mask']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                    output = self.model(image)

                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                self.evaluator.add_batch(target, pred)

        iou, miou = self.evaluator.mean_intersection_over_union()
        print("mIoU:{:.3f}".format(miou))
        print("IoU:{}".format(iou))


def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet-stdconv and UNet-equiconv")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_witdh", type=int, default=512)
    parser.add_argument("--conv_type", type=str, default='Std', choices=['Std', 'Equi'])
    parser.add_argument("--copy_weights", type=str, default=True)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--dataset_path', type=str, default="path to/CVRG-Pano", help='path to dataset')
    args = parser.parse_args()
    test = Test(args)
    test.run()


if __name__ == "__main__":
    main()
