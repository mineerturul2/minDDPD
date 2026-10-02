import json
import os

import click
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from PIL import Image

from model import DDPDConfig, DDPDModel, configure_optimizers, print0, set_seed

MASK_IDX = 0


class ImageTokenDataset(Dataset):
    def __init__(
        self,
        safetensor_path="/home/ubuntu/simo/nano-diffusion-speedrun/tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors",
        debug=False,
    ):
        print(f"Initializing ImageTokenDataset with path: {safetensor_path}")
        self.safetensor_path = safetensor_path

        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        print(f"Loading metadata from: {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.total_samples = self.metadata["total_samples"]
            print(f"Total samples in metadata: {self.total_samples}")

        print(f"Loading tensors from: {safetensor_path}")

        with safe_open(safetensor_path, framework="pt") as f:
            self.indices = f.get_tensor("indices").to(torch.uint16).long()
            self.labels = f.get_tensor("labels").long()
            print(
                f"Loaded indices shape: {self.indices.shape}, labels shape: {self.labels.shape}"
            )

        if debug:
            samplesze = 64
            self.indices = self.indices[:samplesze]
            self.labels = self.labels[:samplesze]
            self.total_samples = samplesze
            print(f"Debug mode: reduced to {samplesze} samples")
            print(self.labels)

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        try:
            # Get indices and reshape to 1D
            indices = self.indices[idx].reshape(-1)
            label = self.labels[idx]
            return indices, label
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            raise


class MNISTTokenDataset(Dataset):
    def __init__(self, debug=False):
        print("Initializing MNISTTokenDataset")
        from torchvision import transforms
        from torchvision.datasets import MNIST

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),  # Resize to 8x8 like the ImageNet dataset
                transforms.Lambda(lambda x: (x * 7.0).long()),  # Scale to 16-bit range
            ]
        )

        self.mnist = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.total_samples = len(self.mnist)
        print(f"Total samples: {self.total_samples}")

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        try:
            image, label = self.mnist[idx]
            # Flatten the 8x8 image into 64 tokens
            indices = image.reshape(-1)
            return indices, label
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            raise


def setup_distributed():
    if dist.is_initialized():
        return

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_step(
    batch,
    planner,
    denoiser,
    grad_accumulation_steps,
):
    device = "cuda"
    indices, labels = batch
    indices = indices.to(device)
    labels = labels.to(device)
    t = torch.rand(indices.shape[0], device=device)

    # Create binary mask based on timestep t
    mask = torch.bernoulli(t.unsqueeze(1).expand(-1, indices.shape[1])).bool()

    # print0(f"mask shape: {mask.shape}, mask: {mask[:3, :3]}")

    # Create corrupted version by cloning original indices
    input_indices = indices.clone()
    # corrupted_as_null = indices.clone()
    # corrupted_as_null[mask] = MASK_IDX

    # Only corrupt tokens where mask is True
    # Sample random tokens from vocab range for corrupted positions
    num_masked = mask.sum().item()
    if num_masked > 0:
        input_indices[mask] = torch.randint(0, MASK_IDX, (num_masked,), device=device)
    corrupted_as_null = input_indices.clone()

    planner_logits, planner_loss = planner(input_indices, t, labels, targets=mask)
    planner_loss = planner_loss / grad_accumulation_steps
    planner_loss.backward()

    # print acc of planner
    with torch.no_grad():
        # since planner is binary classification, we can use accuracy
        where_one = (planner_logits > 0).float()
        where_zero = (planner_logits < 0).float()
        acc = (where_one * mask.float() + where_zero * (1 - mask.float())).mean()
        print0(f"Planner accuracy: {acc:.4f}")

    denoiser_logits, denoiser_loss = denoiser(
        corrupted_as_null, t, labels, targets=indices, mask=mask
    )
    denoiser_loss = denoiser_loss / grad_accumulation_steps
    denoiser_loss.backward()

    with torch.no_grad():
        denoiser_at_mask = denoiser_logits[mask]
        indices_at_mask = indices[mask]
        acc = (denoiser_at_mask.argmax(dim=-1) == indices_at_mask.long()).float().mean()
        print0(f"Denoiser accuracy: {acc:.4f}")

    return {
        "planner_loss": planner_loss.item() * grad_accumulation_steps,
        "denoiser_loss": denoiser_loss.item() * grad_accumulation_steps,
    }


def get_lr_scheduler(optimizer, warmup_iters, lr_decay_iters, max_iters):
    def lr_lambda(step):
        if step < warmup_iters:
            return step / warmup_iters
        else:
            return (max_iters - step) / lr_decay_iters

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


DECODER_PATH = "/home/ubuntu/simo/nano-diffusion-speedrun/tokenize_dataset/pretrained_ckpts/Cosmos-Tokenizer-DI8x8/decoder.jit"


def log_samples(planner, denoiser, device, sequence_length, num_samples=4, mnist=False):
    planner.eval()
    denoiser.eval()
    with torch.no_grad():
        # Sample random class labels for visualization
        if not mnist:
            class_labels = torch.tensor([1, 9, 94, 299], device=device)
        else:
            class_labels = torch.tensor([0, 1, 2, 3], device=device)
        samples = planner.module.sample(
            denoiser.module,
            class_labels=class_labels,
            batch_size=num_samples,
            sequence_length=sequence_length,
            temperature=1.0,
            device=device,
        )
        # Convert samples to text format
        samples_pil = []
        if mnist:
            for sample, label in zip(samples, class_labels):
                # Reshape back to 32x32 for visualization
                sample_2d = sample.reshape(32, 32)
                # to PIL image. make it to 1 channel
                image = sample_2d.cpu().float().numpy() / 8  # shape : 32, 32
                image = Image.fromarray(image, mode="L")
                samples_pil.append(image)
        else:
            decoder = torch.jit.load(DECODER_PATH).to(device)
            tokens = samples.reshape(-1, 32, 32)
            with torch.no_grad():
                decoded_images = decoder(tokens)

            for image in decoded_images:
                image = (image.clamp(-1, 1) + 1) * 127.5
                image = image.permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)
                image = Image.fromarray(image)
                samples_pil.append(image)

    planner.train()
    denoiser.train()
    return samples_pil


@click.command()
@click.option("--batch-size", default=16, help="Batch size per GPU")
@click.option("--planner-lr", default=2e-4, help="Planner learning rate")
@click.option("--denoiser-lr", default=2e-4, help="Denoiser learning rate")
@click.option("--weight-decay", default=0.1, help="Weight decay")
@click.option("--max-iters", default=2000, help="Maximum iterations")
@click.option("--warmup-iters", default=100, help="Warmup iterations")
@click.option("--lr-decay-iters", default=2000, help="LR decay iterations")
@click.option("--grad-clip", default=1.0, help="Gradient clipping")
@click.option(
    "--grad-accumulation-steps", default=1, help="Gradient accumulation steps"
)
@click.option("--log-interval", default=100, help="Log interval")
@click.option("--save-interval", default=1000, help="Save interval")
@click.option("--wandb-project", default="ddpd", help="Weights & Biases project name")
@click.option(
    "--wandb-entity",
    default=None,
    help="Weights & Biases entity (username or team name)",
)
@click.option("--wandb-name", default=None, help="Weights & Biases run name")
@click.option("--mnist", default=True, help="Use MNIST dataset")
@click.option("--ckpt-dir", default="checkpoints", help="Checkpoint directory")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
def train(
    batch_size,
    planner_lr,
    denoiser_lr,
    weight_decay,
    max_iters,
    warmup_iters,
    lr_decay_iters,
    grad_clip,
    grad_accumulation_steps,
    log_interval,
    save_interval,
    wandb_project,
    wandb_entity,
    wandb_name,
    mnist,
    ckpt_dir,
    seed,
):
    # Set seed for reproducibility if provided
    if seed is not None:
        set_seed(seed)

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # Initialize wandb only on rank 0
    if local_rank == 0:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                "batch_size": batch_size,
                "planner_lr": planner_lr,
                "denoiser_lr": denoiser_lr,
                "weight_decay": weight_decay,
                "max_iters": max_iters,
                "warmup_iters": warmup_iters,
                "lr_decay_iters": lr_decay_iters,
                "grad_clip": grad_clip,
                "grad_accumulation_steps": grad_accumulation_steps,
            },
        )

    print(f"Rank {local_rank}: Creating dataset")
    if mnist:
        train_dataset = MNISTTokenDataset(debug=True)
    else:
        train_dataset = ImageTokenDataset(
            safetensor_path="/home/ubuntu/simo/nano-diffusion-speedrun/tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors"
        )
    print(f"Rank {local_rank}: Dataset created successfully")

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
    )
    print(f"Rank {local_rank}: DataLoader created successfully")

    # Initialize models
    config = DDPDConfig(
        vocab_size=int(2**16) + 1,  # ImageNet tokens + 1 for mask.
        block_size=1024,  # 32x32 image tokens
        n_layer=14,
        n_head=6,
        n_embd=768,
    )

    if mnist:
        config.vocab_size = 9
        config.block_size = 1024
        config.num_classes = 10

    global MASK_IDX
    MASK_IDX = config.vocab_size - 1
    print(f"Rank {local_rank}: MASK_IDX: {MASK_IDX}")
    print(f"Rank {local_rank}: Creating models with config: {config}")
    planner = DDPDModel(config, model_type="planner").to(device)
    denoiser = DDPDModel(config, model_type="denoiser").to(device)

    planner = DDP(planner, device_ids=[local_rank], find_unused_parameters=True)
    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=True)
    print(f"Rank {local_rank}: Models created successfully")

    # planner = torch.compile(planner, mode="reduce-overhead")
    # denoiser = torch.compile(denoiser, mode="reduce-overhead")

    # Setup optimizers and schedulers
    planner_optimizer = configure_optimizers(
        planner, weight_decay, planner_lr, (0.95, 0.99), "cuda"
    )
    denoiser_optimizer = configure_optimizers(
        denoiser, weight_decay, denoiser_lr, (0.95, 0.99), "cuda"
    )

    planner_scheduler = get_lr_scheduler(
        planner_optimizer, warmup_iters, lr_decay_iters, max_iters
    )
    denoiser_scheduler = get_lr_scheduler(
        denoiser_optimizer, warmup_iters, lr_decay_iters, max_iters
    )

    # Training loop
    iter_num = 0
    train_iter = iter(train_dataloader)

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    while iter_num < max_iters:
        planner.train()
        denoiser.train()

        planner_optimizer.zero_grad(set_to_none=True)
        denoiser_optimizer.zero_grad(set_to_none=True)

        losses = []
        for _ in range(grad_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_sampler.set_epoch(iter_num)
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            with ctx:
                losses.append(
                    train_step(
                        batch,
                        planner,
                        denoiser,
                        grad_accumulation_steps,
                    )
                )

        # Gradient clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(planner.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)

        planner_optimizer.step()
        denoiser_optimizer.step()

        planner_scheduler.step()
        denoiser_scheduler.step()

        if iter_num % log_interval == 0 and local_rank == 0:
            avg_losses = {
                k: sum(d[k] for d in losses) / len(losses) for k in losses[0].keys()
            }

            lr = planner_scheduler.get_last_lr()[0]

            print(
                f"iter {iter_num}: planner_loss {avg_losses['planner_loss']:.4f}, "
                f"denoiser_loss {avg_losses['denoiser_loss']:.4f}, "
                f"lr {lr:.2e}"
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "iter": iter_num,
                    "planner_loss": avg_losses["planner_loss"],
                    "denoiser_loss": avg_losses["denoiser_loss"],
                    "learning_rate": lr,
                }
            )

        if iter_num % save_interval == 0 and local_rank == 0:
            sample_pil_images = log_samples(
                planner, denoiser, device, 1024, mnist=mnist
            )  # 32x32 = 1024 tokens
            wandb.log(
                {
                    "images": [
                        wandb.Image(img, caption=f"Samples {i}")
                        for i, img in enumerate(sample_pil_images)
                    ]
                }
            )
            checkpoint = {
                "planner_state_dict": planner.module.state_dict(),
                "denoiser_state_dict": denoiser.module.state_dict(),
                "planner_optimizer": planner_optimizer.state_dict(),
                "denoiser_optimizer": denoiser_optimizer.state_dict(),
                "planner_scheduler": planner_scheduler.state_dict(),
                "denoiser_scheduler": denoiser_scheduler.state_dict(),
                "config": {
                    "planner_config": planner.module.config,
                    "denoiser_config": denoiser.module.config,
                },
            }
            save_path = f"{ckpt_dir}/checkpoint_iter_{iter_num}.pt"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"Rank {local_rank}: Checkpoint saved to {save_path}")

        iter_num += 1

    if local_rank == 0:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
