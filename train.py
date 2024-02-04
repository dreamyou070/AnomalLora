import os
import argparse, torch
from model.diffusion_model import load_SD_model
from model.tokenizer import load_tokenizer
from model.lora import LoRANetwork
from data.mvtec import MVTecDRAEMTrainDataset
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
def main(args) :

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    log_dir = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f'\n step 2. model')
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet = load_SD_model(args)
    network = LoRANetwork(text_encoder=text_encoder, unet=unet,
                          lora_dim = args.network_dim, alpha = args.network_alpha,)
    if args.network_weights is not None:
        network.load_weights(args.network_weights)

    print(f'\n step 3. optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    print(f'\n step 4. dataset and dataloader')
    dataset = MVTecDRAEMTrainDataset(root_dir=args.data_path + args.obj_name + "/train/good/",
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512,512],
                                     tokenizer=tokenizer,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=16)

    print(f'\n step 5. lr')
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION['cosine_with_restarts']
    num_training_steps = len(dataloader) * args.num_epochs
    num_cycles = args.lr_scheduler_num_cycles
    lr_scheduler = schedule_func(optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=num_cycles, )


    # dataset.take_text_model(tokenizer, text_encoder)
    #
    # input_ids = batch["input_ids"].to(accelerator.device)
    # encoder_hidden_states = get_hidden_states(args, input_ids, tokenizer, text_encoders, weight_dtype)


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
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec/')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    # step 5. lr
    parser.add_argument('--num_epochs, type=int', default=10)
    parser.add_argument('--lr_scheduler_num_cycles', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=100)

    args = parser.parse_args()
    main(args)
