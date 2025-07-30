'''
Some code modified partially from ESM implementation in Huggingface and DPLM (https://github.com/bytedance/dplm).
---------------------------
Copyright (c) 2025 Institute for AI Industry Research (AIR), Tsinghua University, and AI For Science Group, Shanghai Artificial Intelligence Laboratory
SPDX-License-Identifier: Apache-2.0
'''

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from lightning import LightningModule
from omegaconf import DictConfig
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.models.esm.modeling_esm import (
    EsmSelfAttention, 
    EsmAttention,
    EsmEmbeddings,
    EsmLayer,
    EsmEncoder,
    EsmModel,
    EsmForMaskedLM,
    EsmPreTrainedModel,
    EsmSelfOutput,
    EsmIntermediate,
    EsmOutput,
    EsmContactPredictionHead,
    EsmLMHead,
    EsmPooler
)
import os
import math
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def setup_network(net, load_pretrained_ckpt):
    if isinstance(net, DictConfig):
        assert net.config.get('pretrained_model_name_or_path') is not None
        config = AutoConfig.from_pretrained(**config)
        net = AutoModelForMaskedLM.from_config(config)
    
    if load_pretrained_ckpt is not None:
        if os.path.exists(load_pretrained_ckpt):
            state_dict = torch.load(load_pretrained_ckpt, map_location='cpu')['state_dict']
            net.load_state_dict(state_dict, strict=True)
            del state_dict
        else:
            ptrn_net = AutoModelForMaskedLM.from_pretrained(load_pretrained_ckpt)
            net.load_state_dict(ptrn_net.state_dict(), strict=True)
            del ptrn_net
    return net

def discreteBayesianFlow(t, x, beta1, beta_time_order=2, mask = None):
    """
    Args:
        t: [B, N]
        x: [B, N, K], already one-hot
        beta1: [B, N]
    """
    # last_index = mask.sum(dim=-1).long() -1
    # if mask is not None:
        
    K = x.size(-1)
    beta = beta1 * (t**beta_time_order)  # (B, N)
    beta = beta.unsqueeze(-1)  # (B, N, 1)
    mean = beta * (K * x - 1)  # (B, N, K)
    std = (beta * K).sqrt()  # (B, N, 1)
    eps = torch.randn_like(mean)  # (B, N, K)
    y = mean + std * eps  # (B, N, K)
    theta = F.softmax(y, dim=-1)  # (B, N, K)
    if mask is not None:
        theta = theta * mask[...,None] + (1 - mask[...,None]) * x
    return theta

class ProfileBFNModule(LightningModule):
    def __init__(
        self,
        bfn: Union[nn.Module, DictConfig],
        **kwargs,
    ):
        super().__init__()
        self.bfn = bfn

class ProfileBFN(nn.Module):
    # _default_cfg = DPLMConfig()

    def __init__(self, cfg, net=None, load_pretrained_ckpt=None, **kwargs):
        super().__init__()
        # self._update_cfg(cfg)
        self.cfg = cfg

        self.net = setup_network(net, load_pretrained_ckpt)

        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

    def forward(self, t, inputs_embeds, attention_mask, output_attentions=False, **kwargs):
        outputs = self.net(
            t = t,
            # t = (1.0 - t) * self.cfg.num_diffusion_timesteps,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        logits = outputs["logits"]
        if output_attentions:
            attention_weights = outputs["attention_weights"]
            return logits, attention_weights
        else:
            return logits

    def get_all_hiddens(self, t, inputs_embeds, attention_mask, **kwargs):
        outputs = self.net(
            t = t,
            # t = (1.0 - t) * self.cfg.num_diffusion_timesteps,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs

    def generation(self, inputs_embeds, attention_mask, **kwargs):
        infer_step = int((1 - self.cfg.infer_start) * self.cfg.num_diffusion_timesteps)
        # inputs_embeds.
        mask = attention_mask.clone()
        idx_mask = mask.sum(dim=-1).long() - 1
        mask[torch.arange(mask.shape[0]),idx_mask] = 0
        mask[:,0] = 0
        for idx, _t in enumerate(tqdm(np.linspace(start = self.cfg.infer_start, stop=1.0, num=infer_step + 1)[:-1])):
            t = (torch.ones_like(attention_mask) * _t).to(inputs_embeds)
            if idx > 0:
                inputs_embeds = discreteBayesianFlow(t, probs, beta1=self.cfg.beta1, beta_time_order=self.cfg.beta_time_order, mask = mask)

            
            pred_logits = self.forward(t, inputs_embeds, attention_mask)
            probs = torch.nn.functional.softmax(pred_logits, dim=-1) * mask[..., None] + inputs_embeds * (1 - mask[..., None])
        # probs = probs[...,1:-1,:]
        output_results = [
            "".join(seq.split(" "))
            for seq in self.tokenizer.batch_decode(torch.argmax(probs, dim=-1), skip_special_tokens=True)
        ]
        return output_results
    
    def get_attentions(self, inputs_embeds, attention_mask, **kwargs):
        t = torch.ones_like(attention_mask).to(inputs_embeds)
        _, attn_weights = self.forward(t, inputs_embeds, attention_mask, output_attentions=True)
        attn_weights = torch.stack(attn_weights, dim=1)
        return attn_weights
        
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class FlashEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config):
        self.num_key_value_groups = 1
        self.is_causal = False
        self.config = config
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer * self.attention_head_size**-0.5
        
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError
        
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        

        # context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=attention_mask, scale=1.0)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()~

        attention_interface = eager_attention_forward
        
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0,
            scaling=1.0,
            **kwargs,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
        # outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class FlashEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = FlashEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

class FlashEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FlashEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = FlashEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
class FlashEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList([FlashEsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

class FlashEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = FlashEsmEncoder(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if self.config._attn_implementation is not None and self.config._attn_implementation == 'flash_attention_2':
            extended_attention_mask = attention_mask
        else:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            # encoder_extended_attention_mask = None
            encoder_extended_attention_mask = encoder_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        # print('AAA',output_hidden_states, return_dict)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(encoder_outputs)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class EsmForBFN(EsmForMaskedLM):
    def __init__(self, config, load_pretrained_ckpt = None):
        
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')

        config = AutoConfig.from_pretrained(**config)
        # net = AutoModelForMaskedLM.from_config(config)
        # config.hidden_dropout_prob = dropout
        
        EsmPreTrainedModel.__init__(self, config)
        self.esm = FlashEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        
        self.init_weights()
        # self.apply(init_bert_params)
    
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id['X']
        
        self.contact_head = None
        # self.t_embedding = TimestepEmbedder(hidden_size = config.hidden_size)
        self.t_embedding = nn.Linear(1, config.hidden_size, bias=True)
        self.esm.contact_head = None
        self.esm.embeddings.position_embeddings = None
        self.tokenizer = tokenizer
        self.load_pretrained_ckpt = load_pretrained_ckpt
    
    def forward(self,
                t,
                inputs_embeds,
                attention_mask,
                input_ids=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
            ):
        # attention_mask = input_ids.ne(self.pad_id)
        inputs_embeds = F.linear(inputs_embeds, self.esm.embeddings.word_embeddings.weight.T)
        inputs_embeds = inputs_embeds + self.t_embedding(t[..., None])
        outputs = self.esm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states = output_hidden_states,
            output_attentions = output_attentions
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        result = {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
        if output_attentions:
            result["attention_weights"] = outputs[1]
        if output_hidden_states:
            result["all_hiddens"] = outputs.hidden_states
        return result
    
    def forward_encoder(self, batch, **kwargs):
        return {}
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask
