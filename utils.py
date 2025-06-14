import torch
from einops import rearrange




class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2,alpha=None):
        self.unet_chunk_size = unet_chunk_size
        self.alpha = alpha

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size

            if video_length == 1:
                # former_frame_index = torch.arange(video_length) - 1
                # former_frame_index[0] = 0
                former_frame_index = [0]*video_length
                key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
                key = key[:, former_frame_index]
                key = rearrange(key, "b f d c -> (b f) d c")
                value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
                value = value[:, former_frame_index]
                value = rearrange(value, "b f d c -> (b f) d c")
            elif self.alpha is None:
                # Sparse Attention
                key = rearrange(key, "(b f) d c -> b f d c", f=video_length)

                temp_batchsize, frame_num, figure_num, figure_w = key.shape

                for i in range(1, frame_num):
                    prev_tokens = key[:, :i, :, :].reshape(temp_batchsize, -1, figure_w)
                    total_prev = prev_tokens.shape[1]
                    indices = torch.linspace(0, total_prev - 1, steps=figure_num).long()


                    sampled = prev_tokens[:, indices, :]  # [batch_size, figure_num, figure_w]
                    key[:, i, :, :] = sampled
                key = rearrange(key, "b f d c -> (b f) d c")

                value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
                temp_batchsize, frame_num, figure_num, figure_w = value.shape
                for i in range(1, frame_num):
                    prev_tokens = value[:, :i, :, :].reshape(temp_batchsize, -1, figure_w)

                    total_prev = prev_tokens.shape[1]
                    indices = torch.linspace(0, total_prev - 1, steps=figure_num).long()

                    sampled = prev_tokens[:, indices, :]  # [batch_size, figure_num, figure_w]

                    value[:, i, :, :] = sampled
                value = rearrange(value, "b f d c -> (b f) d c")
            else:
                # ORF Attention
                former_frame_index = [self.alpha]*video_length
                # former_frame_index = [0,1,2,3,4,5,6,7]
                key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
                key = key[:, former_frame_index]
                key = rearrange(key, "b f d c -> (b f) d c")
                value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
                value = value[:, former_frame_index]
                value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states




