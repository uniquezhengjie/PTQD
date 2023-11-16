"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import logging

import numpy as np
import torch.distributed as dist

import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_quantCorrection_imagenet

from quant_scripts.brecq_quant_model import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer
from copy import deepcopy
from imagenet_2012_labels import label_to_name

n_bits_w = 8
n_bits_a = 8

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")  
    model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")
    return model

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--out_dir', default='./generated')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--qdecoder", action='store_true')
    parser.add_argument("--fp32", action='store_true')
    args = parser.parse_args()
    print(args)
    # init ddp
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5696'
    rank=0
    local_rank=0
    # local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
    ## for debug, not use ddp
    # rank=0
    # local_rank=0
    # Setup PyTorch:
    logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
    if args.resume:
        seed = int(time.time())
        torch.manual_seed(seed + rank)
    else:
        torch.manual_seed(0 + rank)

    if args.qdecoder:
        fx_graph_mode_model_file_path = 'quantized_decoder/decoder_fx_graph_mode_quantized_sd.pth'
        quantized_modelFS = torch.jit.load(fx_graph_mode_model_file_path)

    torch.set_grad_enabled(False)
    device = torch.device("cuda", local_rank)

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0

    # Load model:
    model = get_model()
    if args.fp32:
        model_fp32 = deepcopy(model)
        sampler_fp32 = DDIMSampler(model_fp32)
    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    from quant_scripts.quant_dataset import DiffusionInputDataset
    from torch.utils.data import DataLoader

    dataset = DiffusionInputDataset('imagenet_input_20steps_sd.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True) ## each sample is (16,4,32,32)
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device),cali_y[:32].to(device))
        
    # Start calibration
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()

    ckpt = torch.load('quantw{}a{}_ldm_brecq_sd.pth'.format(n_bits_w, n_bits_a), map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.cuda()
    qnn.eval()
    setattr(model.model, 'diffusion_model', qnn)
    sampler = DDIMSampler_quantCorrection_imagenet(model, n_bits_w=n_bits_w, n_bits_a=n_bits_a, correct=True)

    out_path = os.path.join(args.out_dir, f"brecq_w{n_bits_w}a{n_bits_a}_{args.num_samples}steps{ddim_steps}eta{ddim_eta}scale{scale}_1109_sd.npz")
    print("out_path: ",out_path)
    if args.qdecoder:
        out_path_q = os.path.join(args.out_dir, f"brecq_w{n_bits_w}a{n_bits_a}_{args.num_samples}steps{ddim_steps}eta{ddim_eta}scale{scale}_1109_sd_qdecoder.npz")
    if args.fp32:
        out_path_fp32 = os.path.join(args.out_dir, f"brecq_w{n_bits_w}a{n_bits_a}_{args.num_samples}steps{ddim_steps}eta{ddim_eta}scale{scale}_1109_sd_fp32.npz")
    logging.info("sampling...")
    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        all_images = []
        if args.qdecoder:
            all_images_q = []
        if args.fp32:
            all_images_fp32 = []
        all_labels = []
        if args.resume:
            if os.path.exists(out_path):
                ckpt = np.load(out_path)
                all_images = ckpt['arr_0']
                all_labels = ckpt['arr_1']
                assert all_images.shape[0] % args.batch_size == 0, f'Wrong resume checkpoint shape {all_images.shape}'
                all_images = np.split(all_images,
                                      all_images.shape[0] // args.batch_size,
                                      0)
                all_labels = np.split(all_labels,
                                      all_labels.shape[0] // args.batch_size,
                                      0)

                logging.info('successfully resume from the ckpt')
                logging.info(f'Current number of created samples: {len(all_images) * args.batch_size}')
            if args.qdecoder:
                if os.path.exists(out_path_q):
                    ckpt = np.load(out_path_q)
                    all_images_q = ckpt['arr_0']
                    assert all_images_q.shape[0] % args.batch_size == 0, f'Wrong resume checkpoint shape {all_images_q.shape}'
                    all_images_q = np.split(all_images_q,
                                        all_images_q.shape[0] // args.batch_size,
                                        0)
            if args.fp32:
                if os.path.exists(out_path_fp32):
                    ckpt = np.load(out_path_fp32)
                    all_images_fp32 = ckpt['arr_0']
                    assert all_images_fp32.shape[0] % args.batch_size == 0, f'Wrong resume checkpoint shape {all_images_fp32.shape}'
                    all_images_fp32 = np.split(all_images_fp32,
                                        all_images_fp32.shape[0] // args.batch_size,
                                        0)

        generated_num = torch.tensor(len(all_images) * args.batch_size, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)
    n_samples_per_class = args.batch_size
    while generated_num.item() < args.num_samples:
        class_labels = torch.randint(low=0,
                                     high=args.num_classes,
                                     size=(args.batch_size,),
                                     device=device)
        print(class_labels)
        uc = model.get_learned_conditioning(n_samples_per_class * [""])
        datasets = []
        for c in class_labels:
            name = label_to_name(c).split(',')[0]
            datasets.append("This is a photo of a {}".format(name))
        
        for idx, class_label in enumerate(datasets):
            t0 = time.time()
            xc = torch.tensor(n_samples_per_class*[class_labels[idx]]).to(model.device)
            c = model.get_learned_conditioning(n_samples_per_class*[class_label])
            
            if args.fp32:
                samples_ddim_f32, _ = sampler_fp32.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[4, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta) 
                x_samples_ddim_fp32 = model_fp32.decode_first_stage(samples_ddim_f32)
                x_samples_ddim_fp32 = torch.clamp((x_samples_ddim_fp32+1.0)/2.0, 
                                            min=0.0, max=1.0)
                
                x_samples_ddim_fp32 = (x_samples_ddim_fp32 * 255.).clamp(0, 255).to(torch.uint8)
                x_samples_ddim_fp32 = x_samples_ddim_fp32.permute(0, 2, 3, 1)
                samples_fp32 = x_samples_ddim_fp32.contiguous()
                gathered_samples_fp32 = [torch.zeros_like(samples_fp32) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples_fp32, samples_fp32)  # gather not supported with NCCL

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=n_samples_per_class,
                                            shape=[4, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, 
                                            eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                        min=0.0, max=1.0)
            
            x_samples_ddim = (x_samples_ddim * 255.).clamp(0, 255).to(torch.uint8)
            x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
            samples = x_samples_ddim.contiguous()

            t1 = time.time()
            print('throughput : {}'.format(x_samples_ddim.shape[0] / (t1 - t0)))
            
            gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL

            if args.qdecoder:
                x_samples_ddim = quantized_modelFS(samples_ddim.cpu()).cuda()
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                
                x_samples_ddim = (x_samples_ddim * 255.).clamp(0, 255).to(torch.uint8)
                x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
                samples_q = x_samples_ddim.contiguous()
                
                gathered_samples_q = [torch.zeros_like(samples_q) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples_q, samples_q)  # gather not supported with NCCL

            gathered_labels = [
                torch.zeros_like(xc) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, xc)

            if rank == 0:
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                if args.qdecoder:
                    all_images_q.extend([sample.cpu().numpy() for sample in gathered_samples_q])
                if args.fp32:
                    all_images_fp32.extend([sample.cpu().numpy() for sample in gathered_samples_fp32])
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logging.info(f"created {len(all_images) * n_samples_per_class} samples")
                generated_num = torch.tensor(len(all_images) * n_samples_per_class, device=device)
                if args.resume:
                    if generated_num % 32 == 0:
                        arr = np.concatenate(all_images, axis=0)
                        arr = arr[: args.num_samples]
                        if args.qdecoder:
                            arr_q = np.concatenate(all_images_q, axis=0)
                            arr_q = arr_q[: args.num_samples]
                        if args.fp32:
                            arr_fp32 = np.concatenate(all_images_fp32, axis=0)
                            arr_fp32 = arr_fp32[: args.num_samples]
                        label_arr = np.concatenate(all_labels, axis=0)
                        label_arr = label_arr[: args.num_samples]
                        logging.info(f"intermediate results saved to {out_path}")
                        np.savez(out_path, arr, label_arr)
                        if args.qdecoder:
                            np.savez(out_path_q, arr_q, label_arr)
                        if args.fp32:
                            np.savez(out_path_fp32, arr_fp32, label_arr)
                        logging.info(f"finish saved to {out_path}")
                        del arr
                        del label_arr
            torch.distributed.barrier()
            dist.broadcast(generated_num, 0)

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.qdecoder:
            arr_q = np.concatenate(all_images_q, axis=0)
            arr_q = arr_q[: args.num_samples]
        if args.fp32:
            arr_fp32 = np.concatenate(all_images_fp32, axis=0)
            arr_fp32 = arr_fp32[: args.num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)
        if args.qdecoder:
            np.savez(out_path_q, arr_q, label_arr)
        if args.fp32:
            np.savez(out_path_fp32, arr_fp32, label_arr)

    dist.barrier()
    logging.info("sampling complete")


if __name__ == "__main__":
    main()
