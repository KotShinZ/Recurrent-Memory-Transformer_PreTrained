import math
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import PreTrainedModel

from .PreTrainedRMTConfig import PreTrainedRMTConfig

class MemoryCell(torch.nn.Module):
    """Holds memory tensors.
    Replicates memory tensor for each batch size.
    Adds memory tokens to the input tensor and returns that tensor.
    Processes the model output and returns a new memory state.

    Parameters
    ----------
    torch : _type_
        _description_
    """

    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)
        self.config = base_model.config
        
        # token_type_embeddingsの追加
        #self.token_type_embeddings = torch.nn.Embedding(2, getattr(self.model.config, "n_embd", self.model.config.hidden_size))

    def create_memory(self, num_mem_tokens):
        """Randomly initializes an embedding matrix (tensor) for memory tokens and registers it for gradient computation.
           Sets read and write positions for memory tokens.

        Parameters
        ----------
        num_mem_tokens : _type_
            Number of memory tokens.
        """
        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)
        
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
        memory_weights = (
            torch.randn((num_mem_tokens, memory_dim))# * embeddings.weight.data.std()
        )
        
        self.register_parameter(
            "memory", torch.nn.Parameter(memory_weights, requires_grad=True)
        )

    def set_memory(self, input_shape):
        """Replicates memory tensor for each batch size

        Parameters
        ----------
        input_shape : _type_
            _description_

        Returns
        -------
        _type_
            Replicated memory tensor. (batch_size, num_mem_tokens, memory_dim)
        """
        memory = self.memory.repeat(
            input_shape[0], 1, 1
        )  # 　メモリテンソルをバッチサイズ分だけ複製する
        return memory  # (batch_size, num_mem_tokens, memory_dim)

    def forward(self, input_ids, memory_state=None, **kwargs):
        """Performs inference.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input tensor.
        memory_state : torch.Tensor, optional
            Memory tensor, by default None (num_mem_tokens, memory_dim)

        Returns
        -------
        tuple(tuple, torch.Tensor)
            out : tuple
                Model output.
            new_memory_state : torch.Tensor
                New memory state.
        """
        if memory_state is None:
            # メモリテンソルをバッチサイズ分だけ複製する
            memory_state = self.set_memory(input_ids.shape)

        # メモリトークンを入力テンソルに追加し、そのテンソルを返す
        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        out = self.model(**seg_kwargs)
        #print(out)

        # モデルの出力を処理し、新しいメモリ状態を返す
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state

    def process_input(self, input_ids, memory_state, **kwargs):
        """Adds memory tokens to the input tensor and returns that tensor

        Parameters
        ----------
        input_ids : _type_
            Input tensor.
        memory_state : _type_
            Memory tensor.

        Returns
        -------
        _type_
            Input tensor with added memory tokens. (batch_size, seq_len, hidden_size)
        """
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if inputs_embeds.shape[0] != memory_state.shape[0]: # バッチサイズが異なる場合
            memory_state = self.set_memory(inputs_embeds.shape)
        
        # メモリトークンを入力テンソルに追加
        inputs_embeds = torch.cat(
            [memory_state, inputs_embeds, memory_state], dim=1
        ).to(input_ids.device)
        """
        # token_type_idsの生成
        token_type_ids = torch.zeros_like(inputs_embeds[:, :, 0], dtype=torch.long)
        token_type_ids[:, self.num_mem_tokens:-self.num_mem_tokens] = 1

        # token_type_embeddingsの追加と入力の更新
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        inputs_embeds = inputs_embeds + token_type_embeds
        """

        seg_kwargs["input_ids"] = None
        seg_kwargs["inputs_embeds"] = inputs_embeds
        if kwargs.get("attention_mask") is not None:
            seg_kwargs["attention_mask"] = self.pad_attention_mask(
                kwargs["attention_mask"], inputs_embeds.shape
            )
        seg_kwargs["output_hidden_states"] = True
        
        # Positional Embeddings
        pos_mem1 = torch.arange(self.num_mem_tokens, device=input_ids.device)
        pos_mem2 = torch.arange(self.num_mem_tokens, self.num_mem_tokens * 2, device=input_ids.device)
        pos_seg = torch.arange(self.num_mem_tokens * 2, self.num_mem_tokens * 2 + input_ids.shape[1], device=input_ids.device)
        pos = torch.cat([pos_mem1, pos_seg, pos_mem2], dim=0)
        pos = pos.unsqueeze(0).expand(input_ids.shape[0], -1)
        seg_kwargs["position_ids"] = pos
        
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            attention_mask = torch.cat(
                [
                    torch.ones(
                        shape[0], self.num_mem_tokens, device=attention_mask.device
                    ),
                    attention_mask,
                    torch.ones(
                        shape[0], self.num_mem_tokens, device=attention_mask.device
                    ),
                ],
                dim=1,
            )
            return attention_mask

    def compute_logpi(mean, stddev, action):
        a1 =-0.5 * torch.log(2*torch.fill(stddev.shape, math.pi))
        a2 = -torch.log(stddev)
        a3 = -0.5 * (((action - mean) / stddev) ** 2)
        return a1 + a2 + a3

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]
            out["logits"] = model_outputs.logits[
                :, self.num_mem_tokens : -self.num_mem_tokens
            ]

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [
                    lh[:, self.num_mem_tokens : -self.num_mem_tokens]
                    for lh in model_outputs.hidden_states
                ]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs["attentions"]
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state

    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out