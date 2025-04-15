import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from gamengen.models import ActionEmbeddingModel


class GameNGenPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        action_embedding: ActionEmbeddingModel,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            action_embedding=action_embedding,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.context_length = (
            self.unet.config.in_channels // self.unet.config.out_channels - 1
        )

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    def encode_past_frames(
        self, frames: torch.Tensor, context_length: int
    ) -> torch.Tensor:
        frames = rearrange(frames, "b l c h w -> (b l) c h w")
        latents = self.vae.encode(frames).latent_dist.sample()
        latents = rearrange(latents, "(b l) c h w -> b l c h w", l=context_length)
        latents = latents * self.vae.config.scaling_factor
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            1,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        past_frames: torch.Tensor,
        past_actions: torch.Tensor,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.5,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> ImagePipelineOutput | tuple:
        self._guidance_scale = guidance_scale

        batch_size = past_frames.shape[0]
        num_past_frames = past_frames.shape[1]
        height, width = past_frames.shape[-2:]
        device = self._execution_device

        assert num_past_frames == self.context_length, (
            f"num_past_frames must be equal to the context length, got {num_past_frames} and {self.context_length}"
        )

        # 1. Encode past frames
        # context_latents shape -> (batch_size, context_length, num_channels_latents, height, width)
        context_latents = self.encode_past_frames(past_frames, num_past_frames)

        # 2. Encode past actions
        action_embeds = self.action_embedding(past_actions)
        if self.do_classifier_free_guidance:
            action_embeds = torch.cat([action_embeds, action_embeds])

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare latents
        num_channels_latents = context_latents.shape[2]
        # latents shape -> (batch_size, num_past_frames + 1, num_channels_latents, height, width)
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            context_latents.dtype,
            device,
        )
        latents = torch.cat([context_latents, latents], dim=1)

        # 5. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.do_classifier_free_guidance:
                    uncond_latents = latents.clone()
                    uncond_latents[:, :num_past_frames, ...].zero_()

                    latent_model_input = torch.cat([uncond_latents, latents])
                else:
                    latent_model_input = latents

                # Reshape so that context frames are concatenated at the latent channels
                latent_model_input = rearrange(
                    latent_model_input,
                    "b l c h w -> b (l c) h w",
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Predict the noise residual
                # TODO: Implement noise augmentation
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=action_embeds,
                    return_dict=False,
                )[0]

                # Peform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                # Denoise the last frame which is the next frame to be generated
                next_frame_latent = latents[:, -1, :, :, :]
                next_frame_latent = self.scheduler.step(
                    noise_pred, t, next_frame_latent, return_dict=False
                )[0]
                latents[:, -1, :, :, :] = next_frame_latent

                progress_bar.update()

        # 6. Decode and postprocess the latents
        new_frame_latent = latents[:, -1, :, :, :]
        if not output_type == "latent":
            image = self.vae.decode(
                new_frame_latent / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=[True] * image.shape[0]
            )
        else:
            image = new_frame_latent

        if not return_dict:
            return image

        return ImagePipelineOutput(images=image)
