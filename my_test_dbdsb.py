import os
import glob

import numpy as np
import omegaconf
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as transforms

from functools import partial
import matplotlib.pyplot as plt

from bridge.runners.datasets import Celeba
from bridge.runners.config_getters import get_model
from bridge.trainer_dbdsb import IPF_DBDSB
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets, load_paired_colored_mnist

from accelerate import Accelerator
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from cmmd import compute_cmmd, save_ref

# FID calculation
def normalize_tensor(tensor):
    normalized = tensor / 2 + 0.5
    return normalized.clamp_(0, 1)


def to_uint8_tensor(tensor):
    tensor = normalize_tensor(tensor)
    return tensor.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)


def compute_fid_and_ot_cost(true_dataloader, model_input_dataloader, sample_fn):
    # backward loader y -> x

    ot_cost = 0
    fid = FrechetInceptionDistance().to(device)

    size = len(model_input_dataloader.dataset)

    for item in tqdm(iter(true_dataloader)):
        x = item[0]
        fid.update(to_uint8_tensor(x.expand(-1, 3, -1, -1)).to(device), real=True)

    for item in tqdm(iter(model_input_dataloader)):
        y = item[0]
        fake_sample = sample_fn(y.to(device))
        # print(f"y.shape = {y.shape}, fake_sample.shape = {fake_sample.shape}")
        fid.update(to_uint8_tensor(fake_sample.expand(-1, 3, -1, -1)), real=False)

        ot_cost += F.mse_loss(fake_sample.to(device), y.to(device)) * y.shape[0]

    ot_cost = ot_cost / size

    return fid.compute(), ot_cost
import lpips


def compute_lpips_func(img_input, img_output, loss_fn):
    img_input = img_input.clip(-1., 1.)
    img_output = img_output.clip(-1., 1.)
    reconstructed_lpips = loss_fn(img_input, img_output).cpu().numpy()
    return reconstructed_lpips


def get_activations_paired_dataloader(paired_dataloader, sample_fn, model_fid, ref_embed_file, batch_size=50,
                                      dims=2048, device='cpu', mode="features_gt", cost_type='mse'):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model_fid.eval()

    assert mode in ["features_gt", "features_pred"]

    len_dataset = len(paired_dataloader.dataset)

    if batch_size > len_dataset:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        paired_dataloader.batch_size = len_dataset

    pred_arr = np.empty((len_dataset, dims))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    print(f"Compute activations for mode {mode}")

    cost = 0
    size = len(paired_dataloader.dataset)

    loss_fn_alex = lpips.LPIPS(net='alex').double().to(device)

    start_idx = 0

    lpips_arr = []
    cmmd_ar = []

    for (x, y) in tqdm(paired_dataloader):
        # batch = batch.to(device)
        x_0 = x.to(device)
        x_t_1 = y.to(device)

        # first batch calculate LPIPS and save generated pics
        
        with torch.no_grad():
            if mode == "features_pred":
                fake_sample = sample_fn(x)
                # fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args)
                fake_sample_range = to_range_0_1(fake_sample)
                # print(f"min fake sample = {fake_sample_range.min()}, max fake sample = {fake_sample_range.max()}")
                pred = model_fid(fake_sample_range)[0]

                # if compute_lpips:
                #     lpips_val = compute_lpips_func(x_0, fake_sample, loss_fn_alex)
                #     # print(f"lpips val = {lpips_val}, x0 shape = {x_0.shape}, fake_sample = {fake_sample.shape}")
                #     reconstructed_lpips_arr = list(lpips_val)
                #     lpips_arr.extend(reconstructed_lpips_arr)

                if cost_type == 'mse':
                    cost += (F.mse_loss(x_t_1, fake_sample) * x_t_1.shape[0]).item()
                elif cost_type == 'l1':
                    cost += (F.l1_loss(x_t_1, fake_sample) * x_t_1.shape[0]).item()
                elif cost_type == 'lpips':
                    lpips_val = compute_lpips_func(x_0, fake_sample, loss_fn_alex)
                    # print(f"lpips val = {lpips_val}, x0 shape = {x_0.shape}, fake_sample = {fake_sample.shape}")
                    reconstructed_lpips_arr = list(lpips_val)
                    lpips_arr.extend(reconstructed_lpips_arr)

                    print(f'Mean LPIPS cost {np.mean(lpips_arr)}')
                else:
                    raise Exception('Unknown COST')
                
                fake_sample_range = torch.clamp(fake_sample_range, 0, 1) 
                cmmd_ar.append(fake_sample_range.cpu())
            elif mode == "features_gt":
                x_0_range = to_range_0_1(x_0)
                # print(f"min gt sample = {x_0_range.min()}, max fake sample = {x_0_range.max()}")
                pred = model_fid(x_0_range)[0]
                cmmd_ar.append(x_0_range.cpu())

        # If model output is not scalar, apply global spatial average pooling.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    cost = cost / size
    if mode == "features_pred":
        cmmd_metric = compute_cmmd(torch.cat(cmmd_ar, dim = 0), ref_embed_file + ".npy")
    elif mode == "features_gt":
        cmmd_metric = save_ref(torch.cat(cmmd_ar, dim = 0), ref_embed_file)
    torch.cuda.empty_cache()

    if cost_type == 'lpips':
        lpips_arr = np.array(lpips_arr)
        mean_lpips = np.mean(lpips_arr)
        return pred_arr, mean_lpips, cmmd_metric
    else:
        return pred_arr, cost, cmmd_metric

# def get_activations_paired_dataloader(dataloader, sample_fn, model_fid, batch_size=50,
#                                       dims=2048, device='cpu', mode="features_gt", cost_type='mse'):
#     """Calculates the activations of the pool_3 layer for all images.

#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : Batch size of images for the model to process at once.
#                      Make sure that the number of samples is a multiple of
#                      the batch size, otherwise some samples are ignored. This
#                      behavior is retained to match the original FID score
#                      implementation.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations

#     Returns:
#     -- A numpy array of dimension (num images, dims) that contains the
#        activations of the given tensor when feeding inception with the
#        query tensor.
#     """
#     model_fid.eval()

#     assert mode in ["features_gt", "features_pred"]

#     len_dataset = len(dataloader.dataset)

#     if batch_size > len_dataset:
#         print(('Warning: batch size is bigger than the data size. '
#                'Setting batch size to data size'))
#         dataloader.batch_size = len_dataset

#     pred_arr = np.empty((len_dataset, dims))
#     start_idx = 0

#     cost = 0

#     print(f"mode = {mode}")

#     for (x, _) in tqdm(dataloader):
#         # batch = batch.to(device)
#         x = x.to(device)

#         with torch.no_grad():
#             if mode == "features_pred":
#                 fake_sample = sample_fn(x)
#                 fake_sample_range = normalize_tensor(fake_sample)
#             # print(f"min fake sample = {fake_sample_range.min()}, max fake sample = {fake_sample_range.max()}")
#                 pred = model_fid(fake_sample_range)[0]

#                 fake_sample.clamp_(-1, 1)
#                 if cost_type == 'mse':
#                     cost += (F.mse_loss(x, fake_sample) * x.shape[0]).item()
#                 elif cost_type == 'l1':
#                     cost += (F.l1_loss(x, fake_sample) * x.shape[0]).item()
#                 elif cost_type == 'lpips':
#                     import lpips
#                     loss_fn_alex = lpips.LPIPS(net='alex')
#                     loss_fn_alex.to(device)
#                     cost += (loss_fn_alex(x, fake_sample).mean() * x.shape[0]).item()
#                     print('Calculate LPIPS')
#                     print(cost)
#                 else:
#                     raise Exception('Unknown COST')

#             else:
#                 x_0_range = normalize_tensor(x)
#                 # print(f"min gt sample = {x_0_range.min()}, max fake sample = {x_0_range.max()}")
#                 pred = model_fid(x_0_range)[0]
        
#         # If model output is not scalar, apply global spatial average pooling.
#         # This happens if you choose a dimensionality not equal 2048.
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

#         pred = pred.squeeze(3).squeeze(2).cpu().numpy()

#         pred_arr[start_idx:start_idx + pred.shape[0]] = pred

#         start_idx = start_idx + pred.shape[0]

#     cost = cost / len_dataset

#     return pred_arr, cost


def calculate_activation_statistics_paired_dataloader(dataloader, model_fid, ref_embed_file, batch_size=50, dims=2048,
                                                      device='cpu', sample_fn=None, mode="features_gt",
                                                      cost_type='mse'):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- resize      : resize image to this shape

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    # act = get_activations(files, model, batch_size, dims, device, resize)
    act, cost, cmmd_metric = get_activations_paired_dataloader(dataloader, sample_fn, model_fid, ref_embed_file, batch_size,
                                                  dims, device, mode, cost_type)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, cost, cmmd_metric


def compute_statistics_of_path_or_dataloader(path, dataloader, model_fid, ref_embed_file, batch_size, dims, device,
                                             sample_fn, mode, cost_type):
    
    if (path.endswith('.npz') or path.endswith('.npy')) and os.path.exists(path):
        f = np.load(path, allow_pickle=True)
        try:
            m, s = f['mu'][:], f['sigma'][:]
        except:
            m, s = f.item()['mu'][:], f.item()['sigma'][:]
        print(f"\nread stats from {path}\n")
        cost = 0
        _, _, _, cmmd_metric = calculate_activation_statistics_paired_dataloader(dataloader, model_fid, ref_embed_file, batch_size, dims,
                                                                    device, sample_fn, mode, cost_type)
    else:
        print(f"Compute stats for paired dataloader in the mode {mode}")
    
        m, s, cost, cmmd_metric = calculate_activation_statistics_paired_dataloader(dataloader, model_fid, ref_embed_file, batch_size, dims,
                                                                    device, sample_fn, mode, cost_type)
    
    return m, s, cost, cmmd_metric


def forward_sample(runner, sample_net, init_batch_x, init_batch_y, fb, permute=True, num_steps=None,
                   return_last=True, last_denoise=None):
    # sample_net = self.get_sample_net('f')
    sample_net.eval()
    sample_fn = partial(runner.apply_net, net=sample_net, fb=fb)

    with torch.no_grad():
        # self.set_seed(seed=0 + self.accelerator.process_index)
        init_batch_x = init_batch_x.to(runner.device)
        # init_batch_y = init_batch_y.to(self.device)
        # assert not self.cond_final
        x_tot, _, _, _ = runner.langevin.record_langevin_seq(sample_fn, init_batch_x, init_batch_y, fb,
                                                             sample=runner.transfer,
                                                             num_steps=num_steps, last_denoise=last_denoise)
        
        # print(f"x_tot in sample func = {x_tot.shape}")
    
    if permute:
        x_tot = x_tot.permute(1, 0, *list(range(2, len(x_tot.shape))))  # (num_steps, num_samples, *shape_x)

    if return_last:
        return x_tot[-1], runner.num_steps if num_steps is None else num_steps
    else:
        return x_tot, runner.num_steps if num_steps is None else num_steps


def visualize_and_save_results(trajectory_to_visualize_numpy, real_data, input_data, exp_path,
                               num_of_samples_for_trajectories_to_visualize,
                               num_steps, num_steps_to_show):
    fig, ax = plt.subplots(num_of_samples_for_trajectories_to_visualize, num_steps_to_show + 2,
                           figsize=(2 * num_steps_to_show, 2 * 6))

    timesteps_to_show = np.linspace(1, num_steps, num_steps_to_show).astype(np.int32)

    real_data = real_data.permute((0, 2, 3, 1)).numpy()
    input_data = input_data.permute((0, 2, 3, 1)).numpy()
    # print(f"shape trajectory = {trajectory_to_visualize_numpy.shape}, max = {np.max(trajectory_to_visualize_numpy)}")
    # Tprint(f"real_data max = {np.max(real_data)}, input_data max = {np.max(input_data)}")
    
    for k in range(num_of_samples_for_trajectories_to_visualize):
        for j in range(num_steps_to_show + 2):
            ax[0][j].xaxis.tick_top()
            if j == 0:
                ax[k][j].imshow(input_data[k])
                ax[0][j].set_title('Input')
            elif j <= num_steps_to_show:
                index = j - 1
                timestep = timesteps_to_show[index]
                ax[k][j].imshow(trajectory_to_visualize_numpy[timestep - 1, k])

                ax[0][j].set_title(f'T = {timestep}')
            else:
                ax[k][j].imshow(real_data[k])
                ax[0][j].set_title(f'Real data')
            ax[k][j].xaxis.tick_top()
            # ax[k][j].get_xaxis().set_visible(False)
            ax[k][j].set_xticks([])
            ax[k][j].get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.001)
    path_to_save_figures = os.path.join(exp_path,
                                        f'evolution_step.png')
    plt.savefig(path_to_save_figures)
    print(f"saving {path_to_save_figures}")


def visualize_and_save_results(trajectory_to_visualize_numpy, real_data, input_data, exp_path,
                               num_of_samples_for_trajectories_to_visualize,
                               num_steps, num_steps_to_show):
    fig, ax = plt.subplots(num_of_samples_for_trajectories_to_visualize, num_steps_to_show + 2,
                           figsize=(2 * num_steps_to_show, 2 * 6))

    timesteps_to_show = np.linspace(1, num_steps, num_steps_to_show).astype(np.int32)

    real_data = real_data.permute((0, 2, 3, 1)).numpy()
    input_data = input_data.permute((0, 2, 3, 1)).numpy()
    # print(f"shape trajectory = {trajectory_to_visualize_numpy.shape}, max = {np.max(trajectory_to_visualize_numpy)}")
    # Tprint(f"real_data max = {np.max(real_data)}, input_data max = {np.max(input_data)}")
    
    for k in range(num_of_samples_for_trajectories_to_visualize):
        for j in range(num_steps_to_show + 2):
            ax[0][j].xaxis.tick_top()
            if j == 0:
                ax[k][j].imshow(input_data[k])
                ax[0][j].set_title('Input')
            elif j <= num_steps_to_show:
                index = j - 1
                timestep = timesteps_to_show[index]
                ax[k][j].imshow(trajectory_to_visualize_numpy[timestep - 1, k])

                ax[0][j].set_title(f'T = {timestep}')
            else:
                ax[k][j].imshow(real_data[k])
                ax[0][j].set_title(f'Real data')
            ax[k][j].xaxis.tick_top()
            # ax[k][j].get_xaxis().set_visible(False)
            ax[k][j].set_xticks([])
            ax[k][j].get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.001)
    path_to_save_figures = os.path.join(exp_path,
                                        f'evolution_step.png')
    plt.savefig(path_to_save_figures)
    print(f"saving {path_to_save_figures}")
    
parser = argparse.ArgumentParser()
parser.add_argument('--path_to_DSBM', type=str)
parser.add_argument('--data_dataset', type=str)
parser.add_argument('--input_dataset', type=str)
parser.add_argument('--exp_description', type=str)
parser.add_argument('--path_to_save_info', type=str)
parser.add_argument('--label_output', type=str)
parser.add_argument('--cost_type', default='mse', type=str)
parser.add_argument('--mean_match', action='store_true')
parser.add_argument('--last_denoise', action='store_true')
# parser.add_argument('--batch_size', type=int)

parser.add_argument('--imf_iters', action='append', default=None)
parser.add_argument('--fb_eval', action='append', default=None)
parser.add_argument('--nfe', action='append', default=None)

opt = parser.parse_args()

device = 'cuda:0'

# path_to_DSBM = "/cache/selikhanovych/DSBM_AdvBM"
path_to_DSBM = opt.path_to_DSBM
label_output = opt.label_output
# data_dataset = "ColoredMNIST"
data_dataset = opt.data_dataset
# input_dataset = "mnist_2"
input_dataset = opt.input_dataset
cost_type = opt.cost_type
# exp_description = "exp_name=mnist_2_3,first_coupling=ind,first_num_iter=100000,gamma_max=0.034,gamma_min=0.034,method=dbdsb,num_iter=5000,num_steps=30,path_to_save_info="
exp_description = opt.exp_description


# exp_name = exp_description.split(",")[0].split("=")[1]

exp_name = exp_description

print(f"exp_name = {exp_name}")

# path_to_save_info = "/cache/selikhanovych/DSBM_AdvBM/saved_info/bsdm"
path_to_save_info = opt.path_to_save_info
print(f"path_to_save_info = {path_to_save_info}")
seed = 42
if 'minibatch_ot' in exp_description:
    print(f"Use minibatch OT during training!")
    path_to_logs = os.path.join(path_to_DSBM, 'experiments', f"{input_dataset}_{data_dataset}",
                                f"{exp_description}{path_to_save_info},use_minibatch_ot=False", str(seed))
else:
    print(f"Don't use minibatch OT during training!")
    path_to_logs = os.path.join(path_to_DSBM, 'experiments', f"{input_dataset}_{data_dataset}",
                                f"{exp_description}{path_to_save_info}", str(seed))
    
# if path to logs is right then everything else is right too

config_path = os.path.join(path_to_logs, '.hydra', 'config.yaml')

path_to_save_validation = "./evaluation"
os.makedirs(path_to_save_validation, exist_ok=True)
path_to_save_exp_results = os.path.join(path_to_save_validation, exp_name)
os.makedirs(path_to_save_exp_results, exist_ok=True)

args = omegaconf.OmegaConf.load(config_path)
args['LOGGER'] = None

args.mean_match = opt.mean_match
args.last_denoise = opt.last_denoise

accelerator = Accelerator(cpu=args.device == 'cpu', split_batches=True)
accelerator.print('Directory: ' + os.getcwd())

init_ds, final_ds, mean_final, var_final = get_datasets(args)
valid_ds, test_ds = get_valid_test_datasets(args)

runner = IPF_DBDSB(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                   final_cond_model=None, valid_ds=valid_ds, test_ds=test_ds)

num_steps = runner.num_steps
print(f"Model was trained with num_steps = {num_steps}")

model = get_model(args)

ckpt_paths = os.path.join(path_to_logs, 'checkpoints')
# ckpt_path = 'net_f_001_0000000.ckpt' # sample_net_b_001_0100000.ckpt # sample_net_f_010_0005000.ckpt
# ckpt_name = 'sample_net_b_001_0005000.ckpt'
# ckpt_name = "sample_net_b_001_0100000.ckpt"
# ckpt_name = opt.ckpt_name

all_ckpts = sorted(glob.glob(os.path.join(ckpt_paths, "sample_net*")))
# fix all_ckpts and choose certain ones

all_ckpts_n_vals = opt.imf_iters

if all_ckpts_n_vals is None:
    all_ckpts_n_vals = sorted(list(set([int(os.path.basename(ckpt).split("_")[3]) for ckpt in all_ckpts])))
else:
    all_ckpts_n_vals = [int(item) for item in all_ckpts_n_vals]

fb_list = opt.fb_eval

if fb_list is None:
    fb_list = ['f', 'b']


nfe_list = opt.nfe
if nfe_list is None:
    nfe_list = [4, 8, 16, 32, 64, 100, 256, 512, 1024]
else:
    nfe_list = [int(item) for item in nfe_list]
    
print(f'\n\n\n\nIMF iters to evaluate {all_ckpts_n_vals}\n\n\n')
print(f'\n\n\n\nFB list to evaluate {fb_list}\n\n\n')
print(f'\n\n\n\nNFE list to evaluate {nfe_list}\n\n\n')

# fix n vals

# all_ckpts_n_vals = [1, 3, 5, 7, 9, 10, 13, 15, 17, 20]

# all_ckpts_n_vals = [5, 10, 20]

# all_ckpts_n_vals = [5]

print(f"all_ckpts_n_vals = {all_ckpts_n_vals}")

batch_size = 128

if exp_name == "mnist_2_3":
    print(f"Load Colored MNIST data")
    coloring_seed = 42
    test_ds_mnist_2 = load_paired_colored_mnist(target_number=2, train=False, seed=coloring_seed)
    test_ds_mnist_3 = load_paired_colored_mnist(target_number=3, train=False, seed=coloring_seed)

    colored_mnist_2_loader = DataLoader(test_ds_mnist_2, batch_size=batch_size,
                                        shuffle=False,
                                        drop_last=False)
    colored_mnist_3_loader = DataLoader(test_ds_mnist_3, batch_size=batch_size,
                                        shuffle=False,
                                        drop_last=False)

elif 'celeba_male2female' in exp_name:
    print(f"Load Celeba data")
    transform = transforms.Compose([
        transforms.Resize((args.data.image_size, args.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_dir = "/cache/selikhanovych/DDGAN-Bridge/img_align_celeba"
    data_dir = "/cache/datasets/img_align_celeba"
    male_dataset = Celeba(data_dir, train=False, transform=transform, part_test=0.1, mode="male")
    female_dataset = Celeba(data_dir, train=False, transform=transform, part_test=0.1, mode="female")

    print(f"size test dataset male = {len(male_dataset)}")
    print(f"size test dataset female = {len(female_dataset)}")

    male_loader = DataLoader(male_dataset, batch_size=batch_size,
                             shuffle=False,
                             drop_last=False)
    female_loader = DataLoader(female_dataset, batch_size=batch_size,
                               shuffle=False, drop_last=False)

    male_dataset_train = Celeba(data_dir, train=True, transform=transform, part_test=0.1, mode="male")
    female_dataset_train = Celeba(data_dir, train=True, transform=transform, part_test=0.1, mode="female")

    train_batch_size = 32
    male_loader_train = DataLoader(male_dataset_train, batch_size=train_batch_size,
                                   shuffle=False,
                                   drop_last=False)
    female_loader_train = DataLoader(female_dataset_train, batch_size=train_batch_size,
                                     shuffle=False, drop_last=False)

    num_train_iter = 400000

    print(f"size train dataset male = {len(male_dataset_train)}")
    print(f"size train dataset female = {len(female_dataset_train)}")
    print(f"size train dataloader male = {len(male_loader_train)}, num epochs = {num_train_iter/len(male_loader_train)}")
    print(f"size train dataloader female = {len(female_loader_train)}, num_epochs = {num_train_iter/len(female_loader_train)}")

num_steps_to_show = 10

dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model_fid = InceptionV3([block_idx]).to(device)

compute_torchmetrics = False

for fb in fb_list:
    if fb == 'f':
        if exp_name == "mnist_2_3":
            model_input_dataloader = colored_mnist_3_loader
            true_dataloader = colored_mnist_2_loader
        elif 'celeba_male2female' in exp_name:
            model_input_dataloader = male_loader
            true_dataloader = female_loader
        
        label_input_plot = label_output
        label_output_plot = input_dataset

        path_to_save_gt_fid_statistics = os.path.join(path_to_save_exp_results, f"{input_dataset}_gt_statistics.npz")
    else:
        if exp_name == "mnist_2_3":
            model_input_dataloader = colored_mnist_2_loader
            true_dataloader = colored_mnist_3_loader
        elif 'celeba_male2female' in exp_name:
            model_input_dataloader = female_loader
            true_dataloader = male_loader

        label_input_plot = input_dataset
        label_output_plot = label_output

        path_to_save_gt_fid_statistics = os.path.join(path_to_save_exp_results, f"{label_output}_gt_statistics.npz")
    ref_embed_file = os.path.join(path_to_save_exp_results, f"cmmd_gt_statistics_{fb}")
    m1_test_gt, s1_test_gt, _, _ = compute_statistics_of_path_or_dataloader(
        path_to_save_gt_fid_statistics, true_dataloader, model_fid, ref_embed_file, batch_size, dims, device,
        None, mode="features_gt", cost_type=cost_type
    )

    if not os.path.exists(path_to_save_gt_fid_statistics):
        print(f"saving stats for gt test data to {path_to_save_gt_fid_statistics}")
        np.savez(path_to_save_gt_fid_statistics, mu=m1_test_gt, sigma=s1_test_gt)

    for n_val in all_ckpts_n_vals:
        all_ckpts_fb_n_val = sorted(glob.glob(os.path.join(ckpt_paths, f'sample_net_{fb}_{n_val:03}_*.ckpt')))
        if len(all_ckpts_fb_n_val) > 0:
            path_to_save_exp_results_mode_n_val = os.path.join(path_to_save_exp_results, fb, str(n_val))
            os.makedirs(path_to_save_exp_results_mode_n_val, exist_ok=True)
            print(f"evaluation of models in mode = {fb} and n_val = {n_val}")
            training_iters = sorted([int(os.path.basename(ckpt_training).split(".")[0].split("_")[4])
                                     for ckpt_training in all_ckpts_fb_n_val])
            
            # training_iters = [training_iters[-1]]

            fig_fid, ax_fid = plt.subplots(figsize=(12, 12))
            fig_l2_cost, ax_l2_cost = plt.subplots(figsize=(12, 12))
            fig_cmmd, ax_cmmd = plt.subplots(figsize=(12, 12))

            if compute_torchmetrics:
                fid_vals_torchmetrics = []
                l2_cost_vals_torchmetrics = []
            fid_vals_pytorch_fid = []
            l2_cost_vals_pytorch_fid = []
            cmmd_vals_pytorch_fid = []

            training_iter = training_iters[-1]

            for nfe in nfe_list:
            
            # for training_iter in training_iters:
                # ckpt_path = os.path.join(ckpt_paths, ckpt_name)
                ckpt_name = f'sample_net_{fb}_{n_val:03}_{training_iter:07}.ckpt'
                print(f"evaluate ckpt {ckpt_name}")
                print(f"-----------------------------------")
                ckpt_path = os.path.join(ckpt_paths, ckpt_name)

                path_to_save_results = os.path.join(path_to_save_exp_results_mode_n_val, str(training_iter))
                os.makedirs(path_to_save_results, exist_ok=True)

                model.load_state_dict(torch.load(ckpt_path))

                fb = ckpt_name.split("_")[2]

                model = model.to(device)
                model.eval()

                # num_dsbm_nfe = 100

                num_dsbm_nfe = nfe

                sample_fn_metrics = lambda batch: forward_sample(runner, model, batch, None, permute=True, fb=fb,
                                                                 num_steps=num_dsbm_nfe)[0]
                
                sample_fn_visualize = lambda batch: forward_sample(runner, model, batch, None, permute=True, fb=fb,
                                                                   num_steps=num_dsbm_nfe, return_last=False)[0]

                path_to_save_pred_fid_statistics = os.path.join(path_to_save_results, f"pred_statistics_nfe_{nfe}.npz")

                num_of_samples_for_trajectories_to_visualize = min(8, nfe)

                x = next(iter(true_dataloader))[0][:num_of_samples_for_trajectories_to_visualize]
                y = next(iter(model_input_dataloader))[0][:num_of_samples_for_trajectories_to_visualize]
                fake_sample = sample_fn_visualize(y.to(device))  # (num_dsbm_nfe, num_samples, *shape_x)

                fake_sample_visualize = to_uint8_tensor(fake_sample).cpu().permute((0, 1, 3, 4, 2)).numpy()
                y = to_uint8_tensor(y).cpu()
                x = to_uint8_tensor(x).cpu()
                
                visualize_and_save_results(fake_sample_visualize, x, y, path_to_save_results,
                                           num_of_samples_for_trajectories_to_visualize,
                                           nfe, min(10, nfe))

                if compute_torchmetrics:
                    path_to_save_fid_txt_torchmetrics = os.path.join(path_to_save_results, "fid_torchmetrics.txt")
                    path_to_save_cost_txt_torchmetrics = os.path.join(path_to_save_results, "cost_torchmetrics.txt")
                    if not os.path.exists(path_to_save_fid_txt_torchmetrics):
                        
                        fid_test, cost_test = compute_fid_and_ot_cost(true_dataloader, model_input_dataloader,
                                                                      sample_fn_metrics)

                        fid_test = fid_test.cpu().item()
                        cost_test = cost_test.cpu().item()

                        with open(path_to_save_fid_txt_torchmetrics, "w") as f:
                            f.write(f"FID = {fid_test}")

                        print(f"saving FID to {path_to_save_fid_txt_torchmetrics}")

                        with open(path_to_save_cost_txt_torchmetrics, "w") as f:
                            f.write(f"L2 cost = {cost_test}")

                        print(f"saving FID to {path_to_save_cost_txt_torchmetrics}")

                    else:
                        with open(path_to_save_fid_txt_torchmetrics, "r") as f:
                            line = f.readlines()[0]
                            fid_test = float(line.split(" ")[-1])

                        with open(path_to_save_cost_txt_torchmetrics, "r") as f:
                            line = f.readlines()[0]
                            cost_test = float(line.split(" ")[-1])

                    print(f"torchmetrics: fid test = {fid_test}, cost test = {cost_test}")
                    fid_vals_torchmetrics.append(fid_test)
                    l2_cost_vals_torchmetrics.append(cost_test)

                path_to_save_fid_txt_pytorch_fid = os.path.join(path_to_save_results, f"fid_pytorch_fid_nfe_{nfe}.txt")

                #### Comment for resuming of FID and L2 calculations
                path_to_save_pred_fid_statistics = ''


                print(f"Start FID/cost computation!")
                m1_test_pred, s1_test_pred, test_l2_cost, cmmd = compute_statistics_of_path_or_dataloader(
                    path_to_save_pred_fid_statistics, model_input_dataloader, model_fid, ref_embed_file, batch_size, dims, device,
                    sample_fn_metrics, mode="features_pred", cost_type=cost_type)
                
                # fid_value_test = calculate_frechet_distance(m1_test_pred, s1_test_pred, m1_test_gt, s1_test_gt)

                # if not os.path.exists(path_to_save_pred_fid_statistics):
                #     print(f"saving stats for pred on test data to {path_to_save_pred_fid_statistics}")
                #     np.savez(path_to_save_pred_fid_statistics, mu=m1_test_pred, sigma=s1_test_pred)

                # with open(path_to_save_fid_txt_pytorch_fid, "w") as f:
                #     f.write(f"FID = {fid_value_test}")

                # print(f"saving FID to {path_to_save_fid_txt_pytorch_fid}")

                # if not os.path.exists(path_to_save_fid_txt_pytorch_fid):

                fid_value_test = calculate_frechet_distance(m1_test_pred, s1_test_pred, m1_test_gt, s1_test_gt)
                
                if not os.path.exists(path_to_save_pred_fid_statistics):
                    print(f"saving stats for pred on test data to {path_to_save_pred_fid_statistics}")
                    np.savez(path_to_save_pred_fid_statistics, mu=m1_test_pred, sigma=s1_test_pred)

                with open(path_to_save_fid_txt_pytorch_fid, "w") as f:
                    f.write(f"FID = {fid_value_test}")

                print(f"saving FID to {path_to_save_fid_txt_pytorch_fid}")
                # else:
                #     with open(path_to_save_fid_txt_pytorch_fid, "r") as f:
                #         line = f.readlines()[0]
                #         fid_value_test = float(line.split(" ")[-1])

                print(f'pytorch-fid: test FID = {fid_value_test} for ckpt_name = {ckpt_name} for NFE {nfe}')

                path_to_save_cmmd = os.path.join(path_to_save_results, f"cmmd_nfe_{nfe}.txt")

                if not os.path.exists(path_to_save_cmmd):
                    # test_l2_cost = calculate_cost(sample_fn_metrics, model_input_dataloader, cost_type='mse',
                    #                               verbose=True)

                    with open(path_to_save_cmmd, "w") as f:
                        f.write(f"CMMD = {cmmd}")

                    print(f"saving CMMD to {path_to_save_cmmd}")
                else:
                    with open(path_to_save_cmmd, "r") as f:
                        line = f.readlines()[0]
                        cmmd = float(line.split(" ")[-1])

                cmmd_vals_pytorch_fid.append(cmmd)
                
                #############

                # print('Start FID with no postprocess')
                
                # sample_fn_no_postprocess = lambda batch: forward_sample(runner, model, batch, None, permute=True, fb=fb,
                #                                                  num_steps=num_dsbm_nfe, last_denoise=False)[0]
                
                # m1_test_pred_no_post, s1_test_pred_no_post, test_l2_cost_no_post = compute_statistics_of_path_or_dataloader(
                #     path_to_save_pred_fid_statistics, model_input_dataloader, model_fid, batch_size, dims, device,
                #     sample_fn_no_postprocess, mode="features_pred", cost_type=cost_type)

                
                # print(f"Start \nANOTHER\nFID/cost computation!")
                # m1_test_pred_0, s1_test_pred_0, test_l2_cost_0 = compute_statistics_of_path_or_dataloader(
                #     path_to_save_pred_fid_statistics, model_input_dataloader, model_fid, ref_embed_file, batch_size, dims, device,
                #     sample_fn_metrics, mode="features_pred", cost_type=cost_type)
                
                
                # fid_value_postprocess_no_postprocess = calculate_frechet_distance(m1_test_pred, s1_test_pred, m1_test_pred_0, s1_test_pred_0)

                # print(f'FID fid_value_postprocess_no_postprocess {fid_value_postprocess_no_postprocess}')
                
                # fid_value_no_postprocess = calculate_frechet_distance(m1_test_pred_0, s1_test_pred_0, m1_test_gt, s1_test_gt)

                # print(f'FID fid no_postprocess {fid_value_no_postprocess}')
                
                #############
                
                fid_vals_pytorch_fid.append(fid_value_test)
                
                path_to_save_cost_txt_pytorch_fid = os.path.join(path_to_save_results, f"{cost_type}_cost_pytorch_fid_nfe_{nfe}.txt")
                
                if not os.path.exists(path_to_save_cost_txt_pytorch_fid):
                    # test_l2_cost = calculate_cost(sample_fn_metrics, model_input_dataloader, cost_type='mse',
                    #                               verbose=True)

                    with open(path_to_save_cost_txt_pytorch_fid, "w") as f:
                        f.write(f"L2 cost = {test_l2_cost}")

                    print(f"saving L2 cost to {path_to_save_cost_txt_pytorch_fid}")
                else:
                    with open(path_to_save_cost_txt_pytorch_fid, "r") as f:
                        line = f.readlines()[0]
                        test_l2_cost = float(line.split(" ")[-1])
                print(f'DDGAN code test l2 cost = {test_l2_cost} for ckpt_name = {ckpt_name}')

                l2_cost_vals_pytorch_fid.append(test_l2_cost)

                num_of_samples_for_trajectories_to_visualize = min(8, nfe)

                x = next(iter(true_dataloader))[0][:num_of_samples_for_trajectories_to_visualize]
                y = next(iter(model_input_dataloader))[0][:num_of_samples_for_trajectories_to_visualize]
                fake_sample = sample_fn_visualize(y.to(device))  # (num_dsbm_nfe, num_samples, *shape_x)

                fake_sample_visualize = to_uint8_tensor(fake_sample).cpu().permute((0, 1, 3, 4, 2)).numpy()
                y = to_uint8_tensor(y).cpu()
                x = to_uint8_tensor(x).cpu()
                
                visualize_and_save_results(fake_sample_visualize, x, y, path_to_save_results,
                                           num_of_samples_for_trajectories_to_visualize,
                                           nfe, min(10, nfe))

            if compute_torchmetrics:
                fid_vals_torchmetrics = np.array(fid_vals_torchmetrics)
                ax_fid.plot(nfe_list, fid_vals_torchmetrics, label="torchmetrics")

            fid_vals_pytorch_fid = np.array(fid_vals_pytorch_fid)
            ax_fid.plot(nfe_list, fid_vals_pytorch_fid, label="pytorch-fid")
            ax_fid.set_title(f'FID, test, {label_input_plot}->{label_output_plot}, n = {n_val}')
            ax_fid.set_xlabel("NFE")
            ax_fid.set_ylabel("FID")
            ax_fid.legend()
            ax_fid.grid(True)

            path_to_save_figures = os.path.join(path_to_save_exp_results_mode_n_val,
                                                f"test_fid.png")
            fig_fid.savefig(path_to_save_figures)
            print(f"saving fid graph in {path_to_save_figures}")

            l2_cost_vals_pytorch_fid = np.array(l2_cost_vals_pytorch_fid)
            if compute_torchmetrics:
                l2_cost_vals_torchmetrics = np.array(l2_cost_vals_torchmetrics)
                ax_l2_cost.plot(nfe_list, l2_cost_vals_torchmetrics, label="torchmetrics")
            ax_l2_cost.plot(nfe_list, l2_cost_vals_pytorch_fid, label="pytorch-fid")
            ax_l2_cost.set_title(f'L2 cost, test, {label_input_plot}->{label_output_plot}, n = {n_val}')
            ax_l2_cost.set_xlabel("NFE")
            ax_l2_cost.set_ylabel("Cost")
            ax_l2_cost.legend()
            ax_l2_cost.grid(True)

            path_to_save_figures = os.path.join(path_to_save_exp_results_mode_n_val,
                                                f"test_{cost_type}_cost.png")
            fig_l2_cost.savefig(path_to_save_figures)
            print(f"saving fid graph in {path_to_save_figures}")

            cmmd_vals_pytorch_fid = np.array(cmmd_vals_pytorch_fid)

            ax_cmmd.plot(nfe_list, cmmd_vals_pytorch_fid, label="cmmd")
            ax_cmmd.set_title(f'CMMD, test, {label_input_plot}->{label_output_plot}, n = {n_val}')
            ax_cmmd.set_xlabel("NFE")
            ax_cmmd.set_ylabel("CMMD")
            ax_cmmd.legend()
            ax_cmmd.grid(True)

            path_to_save_figures = os.path.join(path_to_save_exp_results_mode_n_val,
                                                f"test_cmmd.png")
            fig_cmmd.savefig(path_to_save_figures)
            print(f"saving fid graph in {path_to_save_figures}")
            