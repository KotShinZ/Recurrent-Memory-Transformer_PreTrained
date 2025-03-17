import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from .MemoryCell import MemoryCell
from .RecurrentWrapper import RecurrentWrapper
from .PreTrainedRMTConfig import PreTrainedRMTConfig


# @register_for_auto_class("AutoModelForCausalLM")
class RecurrentMemoryTransformer(PreTrainedModel):
    """
    Recurrent Memory Transformer Model Class
    A transformer model that processes long context in segments and retains information using memory
    """
    
    config_class = PreTrainedRMTConfig
    auto_model_class = "AutoModelForCausalLM"
    
    # マッピングを定義してAutoクラスが適切なモデルを見つけられるようにする
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    # AUTO_MAPを定義（モデル名からクラスへのマッピング）
    AUTO_MAP = {
        "AutoModelForCausalLM": "RecurrentMemoryTransformer",
    }
    
    def __init__(self, config, base_model=None):
        """
        Initialization
        
        Parameters
        ----------
        config : PreTrainedRMTConfig
            Model configuration
        base_model : PreTrainedModel, optional
            Base transformer model
        """
        super().__init__(config)
        
        # base_modelが指定されていない場合は、configから自動生成
        if base_model is None:
            # ベースモデルのタイプを確認
            if not hasattr(config, "base_model_type"):
                raise ValueError("configにbase_model_typeが指定されていません。RMTの設定にはベースモデルタイプが必要です。")
            base_model_type = config.base_model_type
            
            # ベースモデル用の設定を作成
            base_config = AutoConfig.from_pretrained(base_model_type)
            
            # RMT固有のパラメータを除外してベースモデルの設定を作成
            rmt_specific_params = ['model_type', 'is_memory_all', 'max_n_segments', 'input_seg_len', 
                                  'output_seg_len', 'align', 'num_mem_tokens', 'base_model_type']
            for key, value in config.__dict__.items():
                if key not in rmt_specific_params and not key.startswith('_'):
                    setattr(base_config, key, value)
            
            # ベースモデルを作成
            base_model = AutoModelForCausalLM.from_config(base_config)
        
        # MemoryCellとRecurrentWrapperの初期化
        memory_cell = MemoryCell(base_model, config.num_mem_tokens)
        self.recurrent_wrapper = RecurrentWrapper(
            memory_cell=memory_cell,
            is_memory_all=config.is_memory_all,
            max_n_segments=config.max_n_segments,
            input_seg_len=config.input_seg_len,
            output_seg_len=config.output_seg_len,
            align=config.align
        )
        
    def get_base_model(self):
        """
        Get the base model
        """
        return self.recurrent_wrapper.memory_cell.model
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, labels_mask=None, 
                inputs_embeds=None, output_attentions=None, output_hidden_states=None):
        """
        Forward pass of the model
        
        Parameters
        ----------
        input_ids : torch.Tensor, optional
            Input tensor
        attention_mask : torch.Tensor, optional
            Attention mask
        labels : torch.Tensor, optional
            Label tensor
        labels_mask : torch.Tensor, optional
            Label mask
        inputs_embeds : torch.Tensor, optional
            Input embeddings
        output_attentions : bool, optional
            Whether to output attention weights
        output_hidden_states : bool, optional
            Whether to output hidden states
        """
        forward_kwargs = {}
        if input_ids is not None:
            forward_kwargs["input_ids"] = input_ids
        if labels is not None:
            forward_kwargs["labels"] = labels
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        if labels_mask is not None:
            forward_kwargs["labels_mask"] = labels_mask
        if inputs_embeds is not None:
            forward_kwargs["inputs_embeds"] = inputs_embeds
        if output_attentions is not None:
            forward_kwargs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            forward_kwargs["output_hidden_states"] = output_hidden_states
        
        #forward_kwargs.update(kwargs)
        
        # 通常の順伝播処理
        out = self.recurrent_wrapper.forward(**forward_kwargs)
        """
        # デバッグ出力を削除（または必要に応じてコメント化）
        # print(out["loss"])
        
        # 分散環境で損失が二重計算されないよう、ワールドサイズで割る
        # これは処理済みの場合は不要なので、環境変数などで制御することも可能
        if torch.distributed.is_initialized() and "loss" in out and out["loss"] is not None:
            # 既にDeepSpeedが処理している可能性があるため、確認が必要
            # テスト目的で一時的に追加（実際の環境に合わせて調整が必要）
            # world_size = torch.distributed.get_world_size()
            # out["loss"] = out["loss"] / world_size
            pass
        """
        return out
    
    def generate(self, **kwargs):
        """
        Text generation
        """
        return self.recurrent_wrapper.generate(**kwargs)
    
    def generate_with_tokenizer(self, tokenizer, input_text, **kwargs):
        """
        Text generation using tokenizer
        """
        return self.recurrent_wrapper.generate_with_tokenizer(tokenizer, input_text, **kwargs)
    
    def get_input_embeddings(self):
        """
        Get input embeddings
        """
        return self.get_base_model().get_input_embeddings()
    
    def set_input_embeddings(self, embeddings):
        """
        Set input embeddings
        """
        self.get_base_model().set_input_embeddings(embeddings)
    
    def get_output_embeddings(self):
        """
        Get output embeddings
        """
        return self.get_base_model().get_output_embeddings()
    
    def resize_token_embeddings(self, new_num_tokens):
        """
        Resize token embeddings
        """
        self.get_base_model().resize_token_embeddings(new_num_tokens)
        return self.get_input_embeddings()

RecurrentMemoryTransformer.register_for_auto_class("AutoModelForCausalLM")