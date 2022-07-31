import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from IFRNet.models import IFRNet, IFRNet_L, IFRNet_S


class ImageDataset(Dataset):
    def __init__(self, input_dir: Path, transform=None) -> None:
        super().__init__()
        self.imgs_path = sorted(list(input_dir.rglob('*_00.jpg')))
        self.transform = transform
        
    def get_size(self):
        shape = np.array(Image.open(str(self.imgs_path[0]))).shape
        return (shape[0], shape[1])

    def __len__(self):
        return len(self.imgs_path) - 1
    
    def __getitem__(self, index):
        img0 = Image.open(str(self.imgs_path[index]))
        img1 = Image.open(str(self.imgs_path[index+1]))
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1


def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    print('Video to images: Start...')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f'Original video frame rate: {frame_rate}')
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}_00.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            print('Video to images: Done!')
            print()
            return digit, frame_rate


def save_inter_images(args, input_dir: Path, num_frames: int):
    print('Save inter images: Start...')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device: {device}')

    if args.model_size == 'N':
        print('Select: Normal Model')
        model = IFRNet()
        model.load_state_dict(torch.load('./IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))
    elif args.model_size == 'S':
        print('Select: Small Model')
        model = IFRNet_S()
        model.load_state_dict(torch.load('./IFRNet/checkpoints/IFRNet_small/IFRNet_S_Vimeo90K.pth'))
    elif args.model_size == 'L':
        print('Select: Large Model')
        model = IFRNet_L()
        model.load_state_dict(torch.load('./IFRNet/checkpoints/IFRNet_large/IFRNet_L_Vimeo90K.pth'))
    else:
        raise NameError(f'{args.model_size} is not excepted.')
    model = model.to(device).eval()

    transform = ToTensor()
    dataset = ImageDataset(input_dir, transform)

    dataloader = DataLoader(dataset, args.batch_size)
    resize_trans = Resize(dataset.get_size())

    for i, data in enumerate(tqdm(dataloader)):
        img0, img1 = data
        img0 = img0.to(device)
        img1 = img1.to(device)
        for e in range(1, args.num_interp):
            embt = torch.ones(len(img0), 1, 1, 1).float().to(device) * (i / args.num_interp)
            with torch.no_grad():
                imgt_pred = model.inference(img0, img1, embt)
                imgt_pred = resize_trans(imgt_pred)
            for b in range(len(imgt_pred)):
                num = i * args.batch_size + b
                save_image(imgt_pred[b].cpu(), f'{input_dir}/img_{str(num).zfill(num_frames)}_{int(100*(e / args.num_interp))}.jpg')
    print('Save inter images: Done!')
    print()


def save_video(img_dir: Path, video_name: Path, frame_rate):
    print("Save video: Start...")

    os.makedirs(str(video_name.parent), exist_ok=True)
    
    pic_data = list(img_dir.rglob("*.jpg"))
    pic_data = [str(p) for p in pic_data]
    pic_data.sort()

    img = pic_data[0]
    img = cv2.imread(img)
    size = (img.shape[1], img.shape[0])

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save = cv2.VideoWriter(str(video_name), fourcc, frame_rate, size)

    for i in tqdm(range(len(pic_data))):
        img_path = pic_data[i]
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        save.write(img)
    
    save.release()

    print(f"Save video to {video_name}")
    print("Save video: Done!")
    print()


def main(args):
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    print(f'Time: {now_str}')
    print(f'Slow Motion: x{args.num_interp}')
    print(f'Batch Size: {args.batch_size}')
    print()

    out_dir = Path(args.output_dir)
    output_frames = out_dir / 'images' / now_str

    num_frames, ori_frame_rate = save_all_frames(args.input_video_path, output_frames, 'img', 'jpg')

    save_inter_images(args, output_frames, num_frames)

    out_video_path = out_dir / 'result' / f'{now_str}.mp4'
    save_video(output_frames, out_video_path, frame_rate=ori_frame_rate)

    if not args.remain_img:
        print(f'Delete: {output_frames}')
        shutil.rmtree(output_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_video_path', help='Path to input video.')
    parser.add_argument('--output-dir', '-o', help='Path to output dir.', default='outputs/')
    parser.add_argument('--num-interp', '-i', help='Number of interpolated images (slow motion x).', default=2, type=int)
    parser.add_argument('--batch-size', '-b', help='batch size.', default=1, type=int)
    parser.add_argument('--model-size', '-m', choices=['N', 'S', 'L'], default='N', type=str)
    parser.add_argument('--remain-img', '-r', help='[flag] remain images.', action='store_true')
    args = parser.parse_args()
    main(args)
