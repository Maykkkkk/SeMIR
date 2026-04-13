import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs to train the total model.')

parser.add_argument('--batch_size', type=int,default=6,help="Batch size to use per GPU")

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')

parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

parser.add_argument('--output_path', type=str, default="output/", help='saving restored images')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='saving checkpoint')
parser.add_argument("--wblogger",type=str,default="SeMIR",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default=1, help = "Number of GPUs to use for training")
parser.add_argument("--resume_ckpt_path",type=str, default="", help="resume Checkpoint path")


options = parser.parse_args()

