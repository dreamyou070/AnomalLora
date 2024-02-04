import os
import argparse, torch
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
    network = LoRANetwork(text_encoder=text_encoder, unet=unet,
                          lora_dim = args.network_dim, alpha = args.network_alpha,)
    if args.network_weights is not None:
        network.load_weights(args.network_weights)

    print(f' step 3. optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    print(f' step 4. dataset and dataloader')
    dataset = MVTecDRAEMTrainDataset(args.data_path + args.obj_name + "/train/good/",
                                     args.anomaly_source_path, resize_shape=[512,512])
    ##dataloader = DataLoader(dataset, batch_size=args.bs,shuffle=True, num_workers=16)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int,default=64)
    parser.add_argument('--network_alpha', type=float,default=4)
    parser.add_argument('--network_weights', type=str)
    # 3. optimizer
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # step 4. dataset and dataloader
    parser.add_argument('--data_path', type=str, default=1e-5)
    parser.add_argument('--obj_name', type=str, default=1e-5)
    parser.add_argument('--anomaly_source_path', type=str)
    args = parser.parse_args()
    main(args)
