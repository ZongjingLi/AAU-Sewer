from dataloader import *
from model    import *
from config   import *
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

opt_parser = argparse.ArgumentParser()
opt_parser.add_argument("--source_dir",       default = "/content/MD_KITTI")
opt_parser.add_argument("--target_dir",       default = "/content/MD_KITTI")
opt_parser.add_argument("--update_steps",     default = 5)
opt_parser.add_argument("--transfer_batch",   default = 10)
opt_parser.add_argument("--transfer_samples", default = 100)
opt_parser.add_argument("--visualize_itrs",   default = 30)
opt_parser.add_argument("--tau",              default = 0.07)
opt_parser.add_argument("--omit_portion",     default = 0.3)
opt_parser.add_argument("--density_reduce",   default = 0.6)
opt = opt_parser.parse_args(args = [])