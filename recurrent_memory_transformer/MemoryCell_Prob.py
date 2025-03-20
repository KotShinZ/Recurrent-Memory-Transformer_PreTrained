from .MemoryCell import MemoryCell
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class MemoryCell_Prob(MemoryCell):
    def process_input(self, input_ids, memory_state, **kwargs):
        """メモリトークンを入力テンソルに追加し、そのテンソルを返す

        Parameters
        ----------
        input_ids : _type_
            入力テンソル。
        memory_state : _type_
            メモリテンソル。

        Returns
        -------
        _type_
            メモリトークンを追加した入力テンソル。(batch_size, seq_len, hidden_size)
        """
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if inputs_embeds.shape[0] != memory_state.shape[0]:
            memory_state = self.set_memory(inputs_embeds.shape)
        
        inputs_embeds = torch.cat(
            [memory_state, inputs_embeds, memory_state, memory_state], dim=1 # 平均と分散を追加
        ).to(self.device)

        seg_kwargs["input_ids"] = None
        seg_kwargs["inputs_embeds"] = inputs_embeds
        if kwargs.get("attention_mask") is not None:
            seg_kwargs["attention_mask"] = self.pad_attention_mask(
                kwargs["attention_mask"], inputs_embeds.shape
            )
        seg_kwargs["output_hidden_states"] = True
        return seg_kwargs
    
    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state_mean = model_outputs.hidden_states[-1][:, -self.num_mem_tokens * 2 : -self.num_mem_tokens]
            memory_state_std = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]
            out["logits"] = model_outputs.logits[
                :, self.num_mem_tokens : -self.num_mem_tokens * 2
            ]
            memory_state = memory_state_mean + memory_state_std * torch.randn_like(memory_state_std)

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