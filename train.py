import os
import argparse
from model.diffusion_model import load_SD_model
from model.lora import LoRANetwork
from data.mvtec import MVTecDRAEMTrainDataset
def main(args) :

    print(f' step 1. setting')
    output_dir = args.output_dir
    log_dir = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    print(f' step 2. model')
    text_encoder, vae, unet = load_SD_model(args)
    network = LoRANetwork(text_encoder=text_encoder, unet=unet, lora_dim = args.network_dim, alpha = args.network_alpha,)


    print(f' step 3. dataset')
    #dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/",
    #                                         args.anomaly_source_path,
    #                                     resize_shape=[256, 256])
    ##dataloader = DataLoader(dataset, batch_size=args.bs,shuffle=True, num_workers=16)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    args = parser.parse_args()
    main(args)
