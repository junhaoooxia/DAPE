import argparse
import datetime
import glob
import inspect
import os
import sys
from inspect import Parameter
from typing import Union
import einops
import imageio
import re
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange

from dape.util import (
    exists,
    instantiate_from_config,
    isheatmap,
)
from scripts.sampling.util import chunk, init_sampling, load_video_keyframes, perform_save_locally_video

MULTINODE_HACKS = False


def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param] != Parameter.empty
    }
    return default_args

def get_step_value(folder_name):
    match = re.search(r'step=(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return 0  # return 0 as default

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--no_date",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--projectname",
        type=str,
        default="dape",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--legacy_naming",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="name run based on config file name if true, else by whole path",
    )
    parser.add_argument(
        "--enable_tf32",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12",
    )
    parser.add_argument(
        "--startup",
        type=str,
        default=None,
        help="Startuptime from distributed script",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="log to wandb",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="Wandb entity name string",
    )
    parser.add_argument(
        "--no_base_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="experiment name shown in wandb",
    )
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from",
        )
    default_args = default_trainer_args()
    for key in default_args:
        parser.add_argument("--" + key, default=default_args[key])
    return parser


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print('available "last" checkpoints:')
    print(ckpt)
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print("version confusion but not bad")
            print(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


class SetupCallback(Callback):
    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
        debug,
        ckpt_name=None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: pl.Trainer, pl_module, exception):
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            if self.ckpt_name is None:
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            else:
                ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            # trainer.save_checkpoint(ckpt_path)    # TODO: for fast debugging, I comment this line.

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if (
                    "metrics_over_trainsteps_checkpoint"
                    in self.lightning_config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time

                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_before_first_step=False,
        enable_autocast=True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            elif "video" in k:
                fps = self.log_images_kwargs.get("video_fps", 3)
                video = images[k]
                if self.rescale:
                    video = (video + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                frames = [video[:, :, i] for i in range(video.shape[2])]
                frames = [torchvision.utils.make_grid(each, nrow=4) for each in frames]
                frames = [einops.rearrange(each, "c h w -> 1 c h w") for each in frames]
                frames = torch.clamp(torch.cat(frames, dim=0), min=0.0, max=1.0)
                frames = (frames.numpy() * 255).astype(np.uint8)

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.gif".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                save_numpy_as_gif(frames, path, duration=1 / fps)

                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image only supports WandbLogger currently"
                    wandb.log({f"{split}/{k}": wandb.Video(frames, fps=fps)})
                    # wandb.log({f"{split}/{k}": wandb.Video(frames, fps=fps)}, step=global_step)
            else:
                data_tmp = images[k]
                if data_tmp.ndim == 5:
                    data_tmp = einops.rearrange(data_tmp, "b c t h w -> (b t) c h w")
                nrow = self.log_images_kwargs.get("n_rows", 8)
                grid = torchvision.utils.make_grid(data_tmp, nrow=nrow)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image only supports WandbLogger currently"
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[
                            img,
                        ],
                        step=pl_module.global_step,
                    )

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and
            # batch_idx > 5 and
            self.max_images > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.float32,  # torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()
                    if self.clamp and not isheatmap(images[k]):
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


def save_numpy_as_gif(frames, path, duration=None):
    """
    save numpy array as gif file
    """
    image_list = []
    for frame in frames:
        image = frame.transpose(1, 2, 0)
        image_list.append(image)
    if duration:
        imageio.mimsave(path, image_list, format="GIF", duration=duration, loop=0)
    else:
        imageio.mimsave(path, image_list, format="GIF", loop=0)


@rank_zero_only
def init_wandb(save_dir, opt, config, group_name, name_str, entity_name):
    print(f"setting WANDB_DIR to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = save_dir
    if opt.debug:
        wandb.init(project=opt.projectname, mode="offline", group=group_name)
    else:
        wandb.init(
            project=opt.projectname,
            config=None,
            settings=wandb.Settings(code_dir="./dape"),
            group=group_name,
            name=name_str,
            entity=entity_name,
        )


def _reset_inference_backends():
    """Deterministic inference backends: no TF32, no cudnn benchmark."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _run_inference_loop(model, inference_cfg):
    """The inference loop shared by both the inference-only fast path and
    the post-training inference block. Identical sampling logic either way
    so that post-training inference with 0 training steps produces the
    same output as the fast path on the same model state.
    """
    model = model.eval()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    num_keyframes = inference_cfg.num_keyframes
    H, W = inference_cfg.H, inference_cfg.W
    num_samples = inference_cfg.num_samples
    batch_size = inference_cfg.batch_size
    negative_prompt = inference_cfg.get("negative_prompt", "ugly, low quality")
    add_prompt = inference_cfg.get("add_prompt", "")

    for edit_typ, prompt in dict(inference_cfg.prompt).items():
        prompts = [prompt] * num_samples
        video_paths = [inference_cfg.video_path] * num_samples

        prompts_chunk = list(chunk(prompts, batch_size))
        video_paths_chunk = list(chunk(video_paths, batch_size))

        for idx, (prompts_b, paths_b) in enumerate(zip(prompts_chunk, video_paths_chunk)):
            bs = min(len(prompts_b), batch_size)
            print(f"\nProgress [{edit_typ}]: {idx} / {len(prompts_chunk)}.")
            keyframes_list = []
            for vp in paths_b:
                kf = load_video_keyframes(
                    vp,
                    inference_cfg.original_fps,
                    inference_cfg.target_fps,
                    num_keyframes,
                    (H, W),
                )
                kf = kf.unsqueeze(0)
                kf = rearrange(kf, "b t c h w -> b c t h w").to(device)
                keyframes_list.append(kf)
            keyframes = torch.cat(keyframes_list, dim=0)
            control_hint = keyframes

            batch = {"txt": list(prompts_b), "control_hint": control_hint}
            batch_uc = {
                "txt": [negative_prompt for _ in range(bs)],
                "control_hint": batch["control_hint"].clone(),
            }
            if add_prompt:
                batch["txt"] = [add_prompt + ", " + each for each in batch["txt"]]

            c, uc = model.conditioner.get_unconditional_conditioning(
                batch_c=batch, batch_uc=batch_uc,
            )

            sampling_kwargs = {}
            for k in c:
                if isinstance(c[k], torch.Tensor):
                    c[k], uc[k] = map(lambda y: y[k][:bs].to(device), (c, uc))
            shape = (4, num_keyframes, H // 8, W // 8)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    randn = torch.randn(bs, *shape).to(device)

                    def denoiser(input, sigma, c):
                        return model.denoiser(
                            model.model, input, sigma, c, **sampling_kwargs
                        )

                    sampler = init_sampling(
                        sample_steps=inference_cfg.sample_steps,
                        sampler_name=inference_cfg.sampler_name,
                        discretization_name=inference_cfg.get(
                            "discretization_name", "LegacyDDPMDiscretization"
                        ),
                        guider_config_target="dape.modules.diffusionmodules.guiders.VanillaCFGTV2V",
                        cfg_scale=inference_cfg.cfg_scale,
                    )
                    sampler.verbose = True

                    samples = sampler(denoiser, randn, c, uc=uc)
                    samples = model.decode_first_stage(samples)

            keyframes_vis = (torch.clamp(keyframes, -1.0, 1.0) + 1.0) / 2.0
            samples_vis = (torch.clamp(samples, -1.0, 1.0) + 1.0) / 2.0
            control_hint_vis = (torch.clamp(c["control_hint"], -1.0, 1.0) + 1.0) / 2.0

            save_path = os.path.join(inference_cfg.save_path, edit_typ)
            perform_save_locally_video(
                os.path.join(save_path, "original"),
                keyframes_vis, inference_cfg.target_fps, "mp4", save_grid=False,
            )
            perform_save_locally_video(
                os.path.join(save_path, "result"),
                samples_vis, inference_cfg.target_fps, "mp4", save_grid=False,
            )
            perform_save_locally_video(
                os.path.join(save_path, "control_hint"),
                control_hint_vis, inference_cfg.target_fps, "mp4", save_grid=False,
            )
            print(f"Saved samples to {save_path}.")


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
    torch.set_float32_matmul_precision(precision="medium")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    # ===========================================================
    # INFERENCE-ONLY FAST PATH
    # When --train False is given and the config has an `inference` section:
    #   - DAPE_DISABLE_ADAPTER=1 (skip adapter_dape construction so RNG is
    #     not consumed by adapter Linear/Conv2d/LayerNorm init)
    #   - strip ckpt_path from config so init_from_ckpt is NOT called inside
    #     __init__; load the ckpt afterwards explicitly (matches standalone
    #     order: instantiate -> cpu -> to(cuda) -> load_ckpt -> eval)
    #   - no Lightning Trainer, no data module, no extra setup that would
    #     consume RNG between seed and noise generation
    # ===========================================================
    if not opt.train and opt.base:
        _peek_cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in opt.base],
                                    OmegaConf.from_dotlist(unknown))
        if hasattr(_peek_cfg, "inference"):
            print("=" * 60)
            print("[main.py] INFERENCE-ONLY FAST PATH")
            print("=" * 60)

            # CRITICAL: disable adapter_dape creation BEFORE any DAPE module is
            # imported / instantiated. Otherwise adapter_dape's Linear/Conv2d/
            # LayerNorm parameters consume RNG during model construction and
            # the noise tensor would differ from the pure base model.
            os.environ["DAPE_DISABLE_ADAPTER"] = "1"
            print("[main.py] DAPE_DISABLE_ADAPTER=1 (adapter_dape skipped)")

            # Strip the lightning section (not used)
            _peek_cfg.pop("lightning", OmegaConf.create())

            # Strip ckpt_path from model.params so init_from_ckpt is NOT
            # called during __init__. We load the checkpoint AFTER model
            # construction.
            _model_cfg_for_init = OmegaConf.create(OmegaConf.to_container(_peek_cfg.model, resolve=True))
            _ckpt_path_to_load = None
            if "ckpt_path" in _model_cfg_for_init.params:
                _ckpt_path_to_load = _model_cfg_for_init.params.pop("ckpt_path")

            _reset_inference_backends()

            # Seed FIRST, then build the model.
            inference_seed = _peek_cfg.inference.get("seed", 42)
            seed_everything(inference_seed)
            print(f"[main.py] seeded with {inference_seed}")

            # Build the model without ckpt
            inf_model = instantiate_from_config(_model_cfg_for_init).cpu()

            # Explicitly load the checkpoint with VAE key remap
            if _ckpt_path_to_load is not None:
                print(f"[main.py] loading ckpt from {_ckpt_path_to_load}")
                inf_model.init_from_ckpt(_ckpt_path_to_load)

            inf_model = inf_model.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ).eval()

            # Run the shared inference loop
            _run_inference_loop(inf_model, _peek_cfg.inference)
            sys.exit(0)
    # ===========================================================
    # END INFERENCE-ONLY FAST PATH
    # ===========================================================

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            checkpoint_dir = os.path.join(logdir, "checkpoints")

            # Use the max step checkpoint file
            ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
            ckpt_files.sort(key=get_step_value, reverse=True)
            if ckpt_files:
                ckpt = ckpt_files[0]
                print("use latest checkpoint: {}".format(ckpt))
            else:
                # If no checkpoint files found, use a random initialized model
                print("no checkpoint file found. not resume")
                ckpt = None

        print("#" * 100)
        print(f'Resuming from checkpoint "{ckpt}"')
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            if opt.no_base_name:
                name = ""
            else:
                if opt.legacy_naming:
                    cfg_fname = os.path.split(opt.base[0])[-1]
                    cfg_name = os.path.splitext(cfg_fname)[0]
                else:
                    assert "configs" in os.path.split(opt.base[0])[0], os.path.split(
                        opt.base[0]
                    )[0]
                    cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                        os.path.split(opt.base[0])[0].split(os.sep).index("configs")
                        + 1 :
                    ]  # cut away the first one (we assert all configs are in "configs")
                    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0]
                    cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
                name = "_" + cfg_name
        else:
            name = ""
        if not opt.no_date:
            nowname = now + name + opt.postfix
        else:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        logdir = os.path.join(opt.logdir, nowname)
        print(f"LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    # move before model init, in case a torch.compile(...) is called somewhere
    if opt.enable_tf32:
        # pt_version = version.parse(torch.__version__)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Enabling TF32 for PyTorch {torch.__version__}")
    else:
        print(f"Using default TF32 settings for PyTorch {torch.__version__}:")
        print(
            f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
        )
        print(f"torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")

    if "LOCAL_RANK" in os.environ:
        os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ.get("LOCAL_RANK")
        print("local rank:", os.environ["LOCAL_RANK"])

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # default to gpu
        trainer_config["accelerator"] = "gpu"
        #
        standard_args = default_trainer_args()
        for k in standard_args:
            if getattr(opt, k) != standard_args[k]:
                trainer_config[k] = getattr(opt, k)

        ckpt_resume_path = opt.resume_from_checkpoint

        if not "devices" in trainer_config and trainer_config["accelerator"] != "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                    "project": opt.projectname,
                    "log_model": False,
                    "entity": opt.wandb_entity,
                },
            },
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "testtube",  # hack for sbord fanatics
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb" if opt.wandb else "csv"]
        if opt.wandb:
            # TODO change once leaving "swiffer" config directory
            try:
                group_name = nowname.split(now)[-1].split("-")[1]
            except:
                group_name = nowname
            default_logger_cfg["params"]["group"] = group_name
            init_wandb(
                os.path.join(os.getcwd(), logdir),
                opt=opt,
                group_name=group_name,
                config=config,
                name_str=nowname,
                entity_name=opt.wandb_entity,
            )
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "epoch={epoch:06}-step={step:07}-train_loss={train/loss:.3f}",
                "verbose": True,
                "save_last": False,
                "auto_insert_metric_name": False,
                "save_top_k": -1,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            # default_modelckpt_cfg["params"]["save_top_k"] = -1

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
        # default to ddp if not further specified
        default_strategy_config = {"target": "pytorch_lightning.strategies.DDPStrategy"}

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            default_strategy_config["params"] = {
                "find_unused_parameters": False,
                # "static_graph": True,
                # "ddp_comm_hook": default.fp16_compress_hook  # TODO: experiment with this, also for DDPSharded
            }
        strategy_cfg = OmegaConf.merge(default_strategy_config, strategy_cfg)
        print(
            f"strategy config: \n ++++++++++++++ \n {strategy_cfg} \n ++++++++++++++ "
        )
        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                    "ckpt_name": melk_ckpt_name,
                },
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {"batch_frequency": 1000, "max_images": 4, "clamp": True},
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                },
            },
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
            print(
                "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
            )
            default_metrics_over_trainsteps_ckpt_dict = {
                "metrics_over_trainsteps_checkpoint": {
                    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "params": {
                        "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        "save_top_k": -1,
                        "every_n_train_steps": 10000,
                        "save_weights_only": True,
                    },
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # cmd line trainer args (which are in trainer_opt) have always priority over config-trainer-args (which are in trainer_kwargs)
        trainer_opt = vars(trainer_opt)
        trainer_kwargs = {
            key: val for key, val in trainer_kwargs.items() if key not in trainer_opt
        }
        trainer = Trainer(**trainer_opt, **trainer_kwargs)

        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        # data.setup()
        print("#### Data #####")
        try:
            for k in data.datasets:
                print(
                    f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
                )
        except:
            print("datasets not yet initialized.")

        # configure learning rate
        if "batch_size" in config.data.params:
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        else:
            bs, base_lr = (
                config.data.params.train.loader.batch_size,
                config.model.base_learning_rate,
            )
        if not cpu:
            # add for different device input type
            if isinstance(lightning_config.trainer.devices, int):
                ngpu = lightning_config.trainer.devices
            elif isinstance(lightning_config.trainer.devices, list):
                ngpu = len(lightning_config.trainer.devices)
            elif isinstance(lightning_config.trainer.devices, str):
                ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = min(
                accumulate_grad_batches * ngpu * bs * base_lr, 1e-4
            )
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                if melk_ckpt_name is None:
                    ckpt_path = os.path.join(ckptdir, "last.ckpt")
                else:
                    ckpt_path = os.path.join(ckptdir, melk_ckpt_name)
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        def _probe_weights(tag, m):
            """Diagnostic: print norms of key trainable params to detect whether
            Stage 1 / Stage 2 actually updated weights."""
            adapter_p = None
            norm_p = None
            for name, p in m.named_parameters():
                if adapter_p is None and "adapter_dape.project1.weight" in name:
                    adapter_p = (name, p)
                if norm_p is None and "input_blocks.0.0.weight" not in name and ".norm" in name and "controlnet" not in name and p.ndim >= 1:
                    # grab any non-controlnet norm weight
                    norm_p = (name, p)
                if adapter_p is not None and norm_p is not None:
                    break
            print(f"[probe:{tag}] ", end="")
            if adapter_p is not None:
                n, p = adapter_p
                print(f"adapter({n}) norm={p.detach().float().norm().item():.6f} "
                      f"mean={p.detach().float().mean().item():.6e}  ", end="")
            if norm_p is not None:
                n, p = norm_p
                print(f"norm({n}) norm={p.detach().float().norm().item():.6f} "
                      f"mean={p.detach().float().mean().item():.6e}", end="")
            print()

        # run
        if opt.train:
            _probe_weights("before_stage1", model)
            try:
                trainer.fit(model, data, ckpt_path=ckpt_resume_path)
            except Exception:
                if not opt.debug:
                    melk()
                raise
            _probe_weights("after_stage1", model)

            # ====== DAPE Stage 2: Visual Adapter tuning ======
            if hasattr(model, "set_stage") and getattr(config.model.params, "freeze_model", "") == "normtuning":
                print("=" * 60)
                print("DAPE Stage 2: Visual Adapter tuning")
                print("=" * 60)
                model.set_stage(2)
                stage2_opt = {k: v for k, v in trainer_opt.items()
                              if k not in ("accelerator", "strategy")}
                stage2_opt["max_steps"] = config.get("Stage2Steps", 70)
                stage2_opt["max_epochs"] = -1
                stage2_kwargs = {k: v for k, v in trainer_kwargs.items()
                                 if k != "strategy"}
                stage2_kwargs["strategy"] = instantiate_from_config(strategy_cfg)
                stage2_trainer = Trainer(**stage2_opt, **stage2_kwargs)
                stage2_trainer.fit(model, data)
                _probe_weights("after_stage2", model)

        if not opt.no_test and not trainer.interrupted and hasattr(model, "test_step"):
            trainer.test(model, data)

        # ====== DAPE Inference (post-training; for train+infer single-run) ======
        # NOTE: For inference-only (inference_only.yaml with --train False),
        # the fast path at the top of __main__ handles things and this block
        # is skipped (sys.exit(0) before here).
        #
        # This block runs after training (Stage 1/2) in a single train+infer
        # command, OR after a 0-step no-op training for verification. It
        # reuses _run_inference_loop so the sampling logic is identical to
        # the fast path; only the setup (seed/backend) and model state
        # (adapter present + trained or untrained) differs.
        #
        # We do NOT try to match the fast path byte-exactly here: the
        # trained model carries adapter_dape (DAPE's method), while the
        # fast path disables it. The two scenarios differ by design.
        if hasattr(config, "inference"):
            _reset_inference_backends()
            inference_seed = config.inference.get("seed", 42)
            seed_everything(inference_seed)
            print(f"[post-train inference] re-seeded with {inference_seed}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()
            _probe_weights("before_inference", model)

            _run_inference_loop(model, config.inference)

    except RuntimeError as err:
        if MULTINODE_HACKS:
            import requests
            import datetime
            import os
            import socket

            device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            hostname = socket.gethostname()
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
            print(
                f"ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
                flush=True,
            )
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

        if opt.wandb:
            wandb.finish()
