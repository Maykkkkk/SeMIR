
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from utils.dataset_utils import SeMIRTrainDataset
from src.model import *
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


torch.set_float32_matmul_precision('medium')



class SeMIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SeMIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.inputext = [
            "Gaussian noise with a standard deviation of 15",
            "Gaussian noise with a standard deviation of 25",
            "Gaussian noise with a standard deviation of 50",
            "Rain degradation with rain lines",
            "Hazy degradation with normal haze",
            "Blur degradation with motion blur",
            "Lowlight degradation"
        ]

        self.text_features = None

    def setup(self, stage=None):
        import clip
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        text_tokens = clip.tokenize(self.inputext).to(self.device)
        with torch.no_grad():
            self.text_features = clip_model.encode_text(text_tokens).to(dtype=torch.float32)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        img_id = de_id.tolist()
        text_prompt_list = [self.inputext[idx] for idx in img_id]

        text_code = self.text_features[img_id]

        restored = self.net(degrad_patch, text_code)
        loss = self.loss_fn(restored, clean_patch)

        if batch_idx == 0:
            print("Text prompts used:", text_prompt_list)

        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
        lr = scheduler.get_last_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="SeMIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = SeMIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1, save_last=True)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    model = SeMIRModel()

    trainer = pl.Trainer(max_epochs=opt.epochs,
                         accelerator="gpu",
                         devices=opt.num_gpus,
                         strategy="ddp_find_unused_parameters_true",
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         # precision="16-mixed",
                         gradient_clip_val=1.0)
    trainer.fit(model=model,
                train_dataloaders=trainloader,
                ckpt_path=opt.resume_ckpt_path if os.path.isfile(opt.resume_ckpt_path) else None
                )


if __name__ == '__main__':
    main()