import os
import json
from typing import Type
from transformers import AutoConfig, PretrainedConfig

def register_to_hf_auto_config(
    config_class: Type[PretrainedConfig],
) -> Type[PretrainedConfig]:
    AutoConfig.register(config_class.model_type, config_class)
    return config_class

class PreTrainedRMTConfig(PretrainedConfig):
    """
    Recurrent Memory Transformer configuration class
    """
    
    model_type = "rmt"
    
    auto_map = {
        "AutoModelForCausalLM": "open_r1.rmt.RecurrentMemoryTransofomer.RecurrentMemoryTransformer"
    }
    
    def __init__(
        self,
        base_model_config=None,
        is_memory_all=True,
        max_n_segments=1,
        input_seg_len=512,
        output_seg_len=512,
        align="left",
        num_mem_tokens=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_config = base_model_config
        self.is_memory_all = is_memory_all
        self.max_n_segments = max_n_segments
        self.input_seg_len = input_seg_len
        self.output_seg_len = output_seg_len
        self.align = align
        self.num_mem_tokens = num_mem_tokens
        
        if base_model_config is not None:
            if type(base_model_config) is not dict:
                dict_config: dict = base_model_config.to_dict()
            else:
                dict_config: dict = base_model_config
                
            for key, value in dict_config.items():
                setattr(self, key, value)
            self.base_model_type = dict_config.get("model_type")
            if self.base_model_type is None:
                raise ValueError("base_model_configにmodel_typeが指定されていません。")
            #PreTrainedRMTConfig.model_type = "rmt_" + self.base_model_type  
    """
    def __repr__(self):
        return f"PreTrainedRMTConfig(is_memory_all={self.is_memory_all}, max_n_segments={self.max_n_segments}, " \
               f"input_seg_len={self.input_seg_len}, output_seg_len={self.output_seg_len}, " \
               f"align='{self.align}', num_mem_tokens={self.num_mem_tokens})"
    """