from diffusers import LDMPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler, VQModel
from diffusers.models import UNet2DModel
import torch_pruning as tp
import torch
import torchvision
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
#parser.add_argument("--pruner", type=str, default='taylor', choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning'])
parser.add_argument("--pruner", type=str, default='random', choices=['random', 'magnitude', 'reinit', 'activation'])

#parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

args = parser.parse_args()

batch_size = args.batch_size



def generate_sample_images(pipeline, num_images=10, num_inference_steps_list=[10, 20, 100], num_samples=50000):
    # Ensure the pipeline is on the correct device

    # Iterate over each number of inference steps (10, 20, 100)
    for num_inference_steps in num_inference_steps_list:
        # Create a subfolder for each number of inference steps
        steps_folder = os.path.join(result_folder, f"{num_inference_steps}_steps")
        if not os.path.exists(steps_folder):
            os.makedirs(steps_folder)

        # Generate images in batches, aiming for the total number of images
        num_batches = num_samples // num_images
        for batch_idx in range(num_batches):
            with torch.no_grad():
                # Generate images using the pipeline
                generated_images = pipeline(batch_size=num_images, num_inference_steps=num_inference_steps).images

            # Save each generated image to the corresponding subfolder
            for i, img in enumerate(generated_images):
                # Create a filename for the image
                img_path = os.path.join(steps_folder, f"generated_image_{batch_idx * num_images + i + 1}.png")
                
                # Save the image
                img.save(img_path)
                print(f"Image {batch_idx * num_images + i + 1} saved at: {img_path}")

if __name__=='__main__':
    #dataset = utils.get_dataset(args.dataset)
    #print(f"Dataset size: {len(dataset)}")
    #train_dataloader = torch.utils.data.DataLoader(
    #    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    #)
    #import torch_pruning as tp

    # loading images for gradient-based pruning
    #clean_images = iter(train_dataloader).next()
    #if isinstance(clean_images, (list, tuple)):
    #    clean_images = clean_images[0]
    #clean_images = clean_images.to(args.device)
    #noise = torch.randn(clean_images.shape).to(clean_images.device)

    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    # load all models
    unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
    vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

    # set to cuda
    torch_device = torch.device(args.device) if torch.cuda.is_available() else "cpu"

    unet.to(torch_device)
    vqvae.to(torch_device)
    example_inputs = {'sample': torch.randn(1, unet.in_channels, unet.sample_size, unet.sample_size).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio>0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance() 
        elif args.pruner == 'random' or args.pruner=='reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'activation':
            imp = tp.importance.ActivationImportance()
        else:
            raise NotImplementedError

        ignored_layers = [unet.conv_out]
        ignored_layers = [unet.conv_out]
        from diffusers.models.attention import Attention
        channel_groups = {}
        for m in unet.modules():
            if isinstance(m, Attention):
                channel_groups[m.to_q] = m.heads
                channel_groups[m.to_k] = m.heads
                channel_groups[m.to_v] = m.heads
        
        pruner = tp.pruner.MagnitudePruner(
            unet,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        base_macs, base_params = tp.utils.count_ops_and_params(unet, example_inputs)
        unet.zero_grad()
        unet.eval()
        import random

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        from diffusers.models.resnet import Upsample2D, Downsample2D
        for m in unet.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels == m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(unet, example_inputs)
        print(unet)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
        unet.zero_grad()
        del pruner

        if args.pruner=='reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            reset_parameters(unet)
    
    pipeline = LDMPipeline(
        unet=unet,
        vqvae=vqvae,
        scheduler=scheduler,
    ).to(torch_device)

    pipeline.save_pretrained(args.save_path)

    # if args.pruning_ratio>0:
    #     os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
    #     torch.save(unet, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))
    # with torch.no_grad():
    #     generator = torch.Generator(device=torch_device).manual_seed(0)
    #     images = pipeline(num_inference_steps=100, batch_size=args.batch_size, output_type="numpy").images
    #     os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
    #     torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_pruning.png".format(args.save_path))
    
        
    result_folder = "result_ldm_structured"

    # Ensure the result folder exists
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    generate_sample_images(pipeline, num_images=10, num_inference_steps_list=[10, 20, 100, 50], num_samples=50000)

