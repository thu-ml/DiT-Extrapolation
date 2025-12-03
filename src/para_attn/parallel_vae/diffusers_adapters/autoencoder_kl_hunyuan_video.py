import functools

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLHunyuanVideo
from diffusers.models.autoencoders.vae import DecoderOutput

import para_attn.primitives as DP
from para_attn.parallel_vae import init_parallel_vae_mesh


def send_tensor(tensor, dst, group):
    tensor = tensor.contiguous()
    dist.send_object_list([tensor.shape], dst=dst, group=group)
    dist.send(tensor, dst=dst, group=group)


def recv_tensor(src, group, device=None, dtype=None):
    objects = [None]
    dist.recv_object_list(objects, src=src, group=group)
    t = torch.empty(objects[0], device=device, dtype=dtype)
    dist.recv(t, src=src, group=group)
    return t


def parallelize_vae(vae: AutoencoderKLHunyuanVideo, *, mesh=None):
    mesh = init_parallel_vae_mesh(vae.device.type, mesh=mesh)

    group = DP.get_group(mesh)
    world_size = DP.get_world_size(group)
    rank = DP.get_rank(mesh)

    vae.enable_tiling()

    @functools.wraps(vae.__class__.tiled_encode)
    def new_tiled_encode(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ):
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        if hasattr(self, "tile_sample_min_height"):
            tile_sample_min_height = self.tile_sample_min_height
        else:
            tile_sample_min_height = self.tile_sample_min_size

        if hasattr(self, "tile_sample_min_width"):
            tile_sample_min_width = self.tile_sample_min_width
        else:
            tile_sample_min_width = self.tile_sample_min_size

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        count = 0
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                if count % world_size == rank:
                    tile = x[:, :, :, i : i + tile_sample_min_height, j : j + tile_sample_min_width]
                    tile = self.encoder(tile)
                    tile = self.quant_conv(tile)
                else:
                    tile = None
                row.append(tile)
                count += 1
            rows.append(row)

        if rank == 0:
            count = 0
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    if count % world_size != rank:
                        rows[i][j] = recv_tensor(count % world_size, group, device=x.device, dtype=x.dtype)
                    count += 1
        else:
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    tile = rows[i][j]
                    if tile is not None:
                        send_tensor(tile, 0, group)

        if rank == 0:
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_width)
                    result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
                result_rows.append(torch.cat(result_row, dim=-1))

            enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        else:
            enc = recv_tensor(rank - 1, group, device=x.device, dtype=x.dtype)
        if rank < world_size - 1:
            send_tensor(enc, rank + 1, group)
        return enc

    vae.tiled_encode = new_tiled_encode.__get__(vae)

    @functools.wraps(vae.__class__.tiled_decode)
    def new_tiled_decode(
        self,
        z: torch.Tensor,
        *args,
        return_dict: bool = False,
        **kwargs,
    ):
        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        count = 0
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                if count % world_size == rank:
                    tile = z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile)
                else:
                    decoded = None
                row.append(decoded)
                count += 1
            rows.append(row)

        if rank == 0:
            count = 0
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    if count % world_size != rank:
                        rows[i][j] = recv_tensor(count % world_size, group, device=z.device, dtype=z.dtype)
                    count += 1
        else:
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    decoded = rows[i][j]
                    if decoded is not None:
                        send_tensor(decoded, 0, group)

        if rank == 0:
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_width)
                    result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
                result_rows.append(torch.cat(result_row, dim=-1))

            dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        else:
            dec = recv_tensor(rank - 1, group, device=z.device, dtype=z.dtype)
        if rank < world_size - 1:
            send_tensor(dec, rank + 1, group)

        if not return_dict:
            return (dec,)
        return DecoderOutput(dec, dec)

    vae.tiled_decode = new_tiled_decode.__get__(vae)

    return vae
