import argparse

from segmentation3d.core.seg_train import train
# CUDA_LAUNCH_BLOCKING=1
# export CUDA_LAUNCH_BLOCKING = 1
def main():

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    long_description = "Training engine for 3d medical image segmentation"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='C:\Project\DentalEngine-main\Train/train_config.py', #'C:\Project\DentalEngine-main/segmentation3d/config/train_config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()
