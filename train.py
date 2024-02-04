import os
import argparse
from model.diffusion_model import load_SD_model

def main(args) :

    print(f' step 1. setting')
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    print(f' step 2. model')
    text_encoder, vae, unet, _ = load_SD_model(args)
    print(f' (2.2) lora')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    args = parser.parse_args()
    main()
