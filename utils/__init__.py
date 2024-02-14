import torch
import os
import ast
import argparse

def arg_as_list(arg):
    v = ast.literal_eval(arg)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
    return v

DEFAULT_EPOCH_NAME = "epoch"
EPOCH_FILE_NAME = "{}-{:06d}"
def default_if_none(value, default):
    return default if value is None else value

def get_epoch_ckpt_name(args, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def save_model(args, ckpt_name, unwrapped_nw, save_dtype):
    os.makedirs(args.output_dir, exist_ok=True)
    save_model_base_dir = os.path.join(args.output_dir, "models")
    os.makedirs(save_model_base_dir, exist_ok=True)
    ckpt_file = os.path.join(save_model_base_dir, ckpt_name)
    metadata = {}
    unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata)

def prepare_dtype(args):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32
    return weight_dtype, save_dtype


