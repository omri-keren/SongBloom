
from .base import *
import omegaconf
from ...vae_frontend import AbstractVAE

def pad_to_fix_length(x, max_len, pad_value=0.):
    bsz, seq_len = x.shape[:2]
    if seq_len >= max_len:
        return x[:, :max_len]
    else:
        pad_len = max_len - seq_len
        pad_tensor = torch.full((bsz, pad_len, *x.shape[2:]), pad_value, dtype=x.dtype, device=x.device)
        padded_tensor = torch.cat([x, pad_tensor], dim=1)
        return padded_tensor

class AudioTokenizerConditioner(WaveformConditioner):
    def __init__(self, output_dim, audio_tokenizer, cache=False, max_len=None):
        super().__init__(output_dim, output_dim)
        self.max_len = max_len
        self.use_cache = cache
        
        self.tokenizer = audio_tokenizer
        # breakpoint()
        
        # TODO if cached and not load vae, receive a dict instead
        if isinstance(self.tokenizer, dict):
            self.tokenizer = omegaconf.DictConfig(self.tokenizer)
            self.code_depth = self.tokenizer.channel_dim
            
            
        elif isinstance(self.tokenizer, AbstractVAE):
            self.tokenizer_tp = "vae"
            if self.use_cache:
                self.code_depth = self.tokenizer.channel_dim
            else:
                self.code_depth = 1 # TODO 强制把输入channel设成1了 self.tokenizer.input_channel
            self.output_proj = nn.Identity() if self.output_dim == self.tokenizer.channel_dim \
                            else nn.Linear(self.tokenizer.channel_dim, self.output_dim, bias=False)
                            
        else:
            raise NotImplementedError
        
        
    def forward(self, x: WavCondition):
        wav, lengths, *_ = x
        B = wav.shape[0]
        wav = wav.reshape(B, self.code_depth, -1)
        # print(wav.shape)
        # import torchaudio
        # torchaudio.save("/apdcephfs_cq7/share_1297902/common/erichtchen/shixisheng/cyy/project/music_generation_repo/core/models/musicgen/conditioners/111.wav", wav[0].cpu(), 48000)
        if self.tokenizer_tp == "vae":
            if self.use_cache:
                audio_latents = wav.transpose(-1,-2)
            else:
                with torch.no_grad():
                    fusion_method = x.fusion_method[0]
                    if fusion_method == '' or fusion_method == 'concat_wav':
                        # audio_latents = self.tokenizer.encode(wav).transpose(-1,-2)
                        audio_latents = self.tokenizer.encode(wav[0].unsqueeze(0)).transpose(-1,-2)
                    else:
                        N = wav.shape[0] - 1 # since there is null condition
                        # TODO: think what to do with null conditioner
                        # Compute embedding for each wav
                        wav_embeds = []
                        for i in range(N):
                            wav_i = wav[i].unsqueeze(0)
                            embed_i = self.tokenizer.encode(wav_i)  # [1, D, T']
                            wav_embeds.append(embed_i)
                        wav_embeds = [e.squeeze(0) for e in wav_embeds]  # [D, T'] each
                        # Stack to [N, D, T']
                        wav_embeds = torch.stack(wav_embeds, dim=0)

                        if fusion_method == 'average':
                            fused = wav_embeds.mean(dim=0, keepdim=True)
                        elif fusion_method == 'product':
                            fused = wav_embeds.prod(dim=0, keepdim=True)
                        elif fusion_method == 'min':
                            fused, _ = wav_embeds.min(dim=0, keepdim=True)
                        elif fusion_method == 'max':
                            fused, _ = wav_embeds.max(dim=0, keepdim=True)
                        elif fusion_method == 'concat_embed':
                            # Concatenate the embeddings (first 1/N of each embedding)
                            D, T_total = wav_embeds.shape[1], wav_embeds.shape[2]
                            T_each = T_total // N
                            segs = []
                            for i in range(N):
                                T_i = wav_embeds.shape[2]
                                if T_i < T_each:
                                    seg = torch.nn.functional.pad(wav_embeds[i], (0, T_each-T_i))
                                else:
                                    seg = wav_embeds[i][:, :T_each]
                                segs.append(seg)
                            fused = torch.cat(segs, dim=1)
                            if fused.shape[1] > T_total:
                                fused = fused[:, :T_total]
                            elif fused.shape[1] < T_total:
                                fused = torch.nn.functional.pad(fused, (0, T_total-fused.shape[1]))
                            fused = fused.unsqueeze(0)
                        audio_latents = fused.transpose(-1, -2)
                    # print('transform wav to vae')
            audio_latents = self.output_proj(audio_latents)

        # print(audio_latents.shape)
        if self.max_len is not None:
            audio_latents = pad_to_fix_length(audio_latents, self.max_len, 0.)
                    
        if lengths is not None:
            lengths = torch.round(lengths.float() * audio_latents.shape[1] / wav.shape[-1])
            mask = length_to_mask(lengths, max_len=audio_latents.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones((B, audio_latents.shape[1]), device=audio_latents.device,dtype=torch.int)

        audio_latents = audio_latents * mask[..., None]
        
        return audio_latents, mask
     

