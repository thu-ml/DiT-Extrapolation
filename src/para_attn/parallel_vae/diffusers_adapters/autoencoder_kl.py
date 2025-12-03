import functools

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
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


def parallelize_vae(vae: AutoencoderKL, *, mesh=None):
    mesh = init_parallel_vae_mesh(vae.device.type, mesh=mesh)

    group = DP.get_group(mesh)
    world_size = DP.get_world_size(group)
    rank = DP.get_rank(mesh)

    vae.enable_tiling()

    @functools.wraps(vae.__class__._tiled_encode)
    def new__tiled_encode(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        count = 0
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                if count % world_size == rank:
                    tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                    tile = self.encoder(tile)
                    if self.config.use_quant_conv:
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
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[:, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=3))

            enc = torch.cat(result_rows, dim=2)
        else:
            enc = recv_tensor(rank - 1, group, device=x.device, dtype=x.dtype)
        if rank < world_size - 1:
            send_tensor(enc, rank + 1, group)
        return enc

    vae._tiled_encode = new__tiled_encode.__get__(vae)

    @functools.wraps(vae.__class__.tiled_decode)
    def new_tiled_decode(
        self,
        z: torch.Tensor,
        *args,
        return_dict: bool = False,
        **kwargs,
    ):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        count = 0
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                if count % world_size == rank:
                    tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                    if self.config.use_post_quant_conv:
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
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[:, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=3))

            dec = torch.cat(result_rows, dim=2)
        else:
            dec = recv_tensor(rank - 1, group, device=z.device, dtype=z.dtype)
        if rank < world_size - 1:
            send_tensor(dec, rank + 1, group)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    vae.tiled_decode = new_tiled_decode.__get__(vae)

    return vae
