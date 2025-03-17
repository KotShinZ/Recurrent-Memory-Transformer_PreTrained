import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .PreTrainedRMTConfig import PreTrainedRMTConfig
from .MemoryCell import MemoryCell
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel

class RecurrentWrapper(torch.nn.Module):
    #config_class = PreTrainedRMTConfig
    
    def __init__(
        self, 
        memory_cell: MemoryCell, 
        is_memory_all: bool, 
        max_n_segments: int, 
        input_seg_len: int, 
        output_seg_len: int, 
        align: str = "left"):
        
        super().__init__()
        self.memory_cell:MemoryCell = memory_cell
        self.is_memory_all = is_memory_all # Whether to share memory state between segments
        self.memory_state: torch.Tensor = None # Memory state
        self.config = memory_cell.config # Model configuration
        self.max_n_segments = max_n_segments # Maximum number of segments for backpropagation
        self.input_seg_len = input_seg_len # Segment size
        self.output_seg_len = output_seg_len
        self.align = align # Segment alignment default: left

    def forward(
        self,
        input_ids,
        labels=None,
        labels_mask=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        """Performs inference.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input tensor. (batch_size, seq_len * n_segments)
        labels : _type_, torch.Tensor
           Input tensor. (batch_size, seq_len * n_segments)

        Returns
        ----------
        dict
            "loss" : torch.Tensor
                Loss value.
            "logits" : torch.Tensor
                Model output.
            "out[f"{key}_{seg_num}"]" : torch.Tensor
                Output for each segment.
        """
        if self.memory_state is not None:
            if self.is_memory_all is False:
                self.memory_state = None
            else :
                self.memory_state.detach()  # メモリ状態の勾配を計算しないようにする

        # 入力テンソルをセグメント単位に分割する。 (セグメントは1ステップでモデルに渡される入力のサブセット)
        segmented = self.segment(
            self.input_seg_len,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        cell_outputs = []  # 各セグメントの出力を保存するリスト
        for seg_num, segment in enumerate(segmented):
            cell_out, self.memory_state = self.memory_cell(
                **segment, memory_state=self.memory_state, **kwargs
            )
            cell_outputs.append(cell_out)
            a = self.manage_gradients(
                self.memory_state, seg_num, len(segmented)
            )  # メモリ状態の勾配計算を制御する
            #print(seg_num, a)

        out = self.process_outputs(
            cell_outputs,
            labels=labels,
            labels_mask=labels_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return out

    def log(self, t, eps = 1e-20):
        return torch.log(t.clamp(min = eps))

    def gumbel_noise(self, t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -self.log(-self.log(noise))

    def gumbel_sample(self, t, temperature = 1., dim = -1):
        return ((t / max(float(temperature), float(1e-10))) + self.gumbel_noise(t)).argmax(dim = dim)

    def top_k(self, logits, thres = 0.9):
        k = math.ceil((1 - thres) * logits.shape[-1])
        val, ind = torch.topk(logits, k)
        probs = torch.full_like(logits, float('-inf'))
        probs.scatter_(1, ind, val)
        return probs

    def segment(self, seg_len, **kwargs):
        """
        Segments input tensors and adjusts their size. Returns a list of dicts.

        Parameters
        ----------
        **kwargs : dict
            Tensors to be segmented.
            Specify tensors that need to be split in keyword argument format.
            Example: segment(input_ids=tensor1, attention_mask=tensor2)

        Returns
        -------
        segments : list of dict
            List of dictionaries containing segmented tensors.
            Example: [{'input_ids': segment1, 'attention_mask': segment1}, {'input_ids': segment2, 'attention_mask': segment2}, ...]

        Notes
        -----
        - This function uses the `self.split_tensor` method, so `self` must implement it.
        - Each tensor is split in a specific way by `self.split_tensor`. The same keys are stored with the same order of indices.
        """
        segments = []  # 各セグメントを保存するリストを初期化
        for k, tensor in kwargs.items():  # keyで繰り返し
            if tensor is not None:
                k_segments = self.split_tensor(
                    tensor, seg_len
                )  # 2次元テンソルを分割し、セグメント化
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg}) # 新たな辞書 {k: k_seg} を作成し、segments リストに追加します。

        return segments

    def split_tensor(self, tensor, seg_len):
        if self.align in {"left", None}:
            split_inds = list(range(0, tensor.shape[1], seg_len)) + [
                tensor.shape[1]
            ]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif self.align in {"right", None}:
            split_inds = (list(range(tensor.shape[1], 0, -seg_len)) + [0])[::-1]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif self.align == "center":
            n_seg = math.ceil(tensor.shape[1] / seg_len)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            split_inds = list(range(0, tensor.shape[1], seg_len)) + [
                tensor.shape[1]
            ]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        """Calculates loss for a list of outputs. Also concatenates and returns logits.

        Parameters
        ----------
        cell_outputs : list of torch.Tensor
            List containing outputs from each segment.

        Returns
        -------
        dict
            "loss" : torch.Tensor
                Loss value.
            "logits" : torch.Tensor
                Model output.
            "out[f"{key}_{seg_num}"]" : torch.Tensor
                Output for each segment.
        """
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat(
            [o.logits for o in cell_outputs], dim=1
        )  # セグメントごとのlogitsを結合する (batch_size, seq_len * seg_len, vocab_size)

        if kwargs.get("output_hidden_states"):
            full_hidden_states = tuple(
                [
                    torch.cat(layer_hs, dim=1)
                    for layer_hs in zip(*[o.hidden_states for o in cell_outputs])
                ]
            )

        labels = kwargs.get("labels")
        if labels is not None:  # ラベルがある場合のみlossを計算する
            
            shift_labels = labels[..., 1:].contiguous() # DataSetでシフトされない場合
            shift_logits = full_logits[..., :-1, :].contiguous()#  DataSetでシフトされない場合
            #shift_labels = labels.contiguous() # DataSetでシフトされる場合
            #shift_logits = full_logits.contiguous() # DataSetでシフトされる場合
            
            flat_labels = shift_labels.view(
                -1
            )  # バッチとセグメントの次元を結合して1次元にする (batch_size * (seq_len-1) * seg_len)
            flat_logits = shift_logits.view(
                -1, shift_logits.size(-1)
            )  # バッチとセグメントの次元を結合して1次元にする (batch_size * (seq_len-1) * seg_len, vocab_size)

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get("labels_mask")
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
            out["loss"] = loss_fct(flat_logits, flat_labels)
        else:
            out["loss"] = 0
            print("labels is None")

        out["logits"] = full_logits
        segment_keys = ["loss", "logits"]
        if kwargs.get("output_attentions"):
            segment_keys.append("attentions")
        if kwargs.get("output_hidden_states"):
            segment_keys.append("hidden_states")
            out["hidden_states"] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f"{key}_{seg_num}"] = value

        return out

    def manage_gradients(self, memory_state, seg_num, seg_len):
        """Controls gradient calculation for memory state

        Parameters
        ----------
        memory_state : torch.Tensor
            Memory state. (batch_size, num_mem_tokens, memory_dim)
        seg_num : int
            Number of the segment currently being processed.

        Returns
        ----------
        bool
            Whether to calculate gradients. True: calculate gradients, False: do not calculate gradients
        """

        # max_n_segments: 処理できる最大セグメント数を示すパラメータです。この値を使って、必要に応じてメモリの更新を決定します。

        # seg_numが0の時はReccurentでない時なので勾配は計算する。
        # 最後のほうのセグメントは勾配を計算する。
        if seg_num == 0 or self.max_n_segments in {-1, None} or seg_len - seg_num <= self.max_n_segments:
            self.memory_state = memory_state  # Retain gradients
            return True
        else:
            self.memory_state = memory_state.detach()  # Detach to stop gradient tracking
            return False

    def generate_groq(
        self,
        input_ids,
        max_length=25,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=True,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        """
        Generate new tokens based on the input sequence.

        Parameters
        ----------
        input_ids : torch.Tensor
            Initial input sequence. Shape: (batch_size, seq_len)
        max_length : int
            Maximum number of tokens to generate (including initial sequence length).
        temperature : float, default 1.0
            Temperature parameter for sampling. Lower values make it more deterministic.
        top_k : int, optional
            Used to sample from top k tokens.
        top_p : float, optional
            Used to filter tokens based on cumulative probability p.
        do_sample : bool, default True
            If True, use probabilistic sampling. If False, use greedy decoding.
        pad_token_id : int, optional
            ID of the padding token.
        eos_token_id : int, optional
            ID of the end-of-sequence token.
        **kwargs : dict
            Additional arguments passed to MemoryCell.

        Returns
        -------
        torch.Tensor
            Generated token sequence. Shape: (batch_size, generated_seq_len)
        """
        # 初期の入力シーケンスを処理
        segmented = self.segment(self.input_seg_len, input_ids=input_ids)
        memory_state = None
        for segment in segmented:
            cell_out, memory_state = self.memory_cell(
                **segment, memory_state=memory_state, **kwargs
            )

        # 生成ループ
        output_ids = input_ids
        while output_ids.shape[1] < max_length:
            # 最後のトークンを input_ids として使用
            last_token = output_ids[:, -1:]
            # MemoryCell に渡す
            cell_out, memory_state = self.memory_cell(
                input_ids=last_token, memory_state=memory_state, **kwargs
            )
            # logits を取得（最後のトークンの logits）
            logits = cell_out.logits[:, -1, :]
            # 次のトークンをサンプリング
            next_token = self.sample_next_token(
                logits, temperature, top_k, top_p, do_sample
            )
            # 出力シーケンスに追加
            output_ids = torch.cat([output_ids, next_token], dim=1)
            # 終了条件をチェック
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return output_ids

    def sample_next_token(self, logits, temperature=1, top_k=50, top_p=0.9, do_sample=False):
        """
        logits から次のトークンをサンプリングする。

        Parameters
        ----------
        logits : torch.Tensor
            トークンの予測スコア。形状: (batch_size, vocab_size)
        temperature : float
            サンプリング時の温度パラメータ。
        top_k : int, optional
            上位 k トークンからサンプリングする場合に使用。
        top_p : float, optional
            累積確率 p に基づいてトークンをフィルタリングする場合に使用。
        do_sample : bool
            True の場合、確率的サンプリングを使用。False の場合、貪欲法を使用。

        Returns
        -------
        torch.Tensor
            サンプリングされたトークン。形状: (batch_size, 1)
        """
        if do_sample:
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                logits = self.top_k_groq(logits, top_k)
            if top_p is not None:
                logits = self.top_p(logits, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token

    def top_k_groq(self, logits, k):
        """
        上位 k トークンのみを考慮するように logits をフィルタリングする。

        Parameters
        ----------
        logits : torch.Tensor
            トークンの予測スコア。形状: (batch_size, vocab_size)
        k : int
            上位 k トークンを選択。

        Returns
        -------
        torch.Tensor
            フィルタリングされた logits。形状: (batch_size, vocab_size)
        """
        values, indices = torch.topk(logits, k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
        return torch.where(
            logits >= min_values, logits, torch.full_like(logits, float('-inf'))
        )

    def top_p(self, logits, p):
        """
        累積確率 p に基づいてトークンをフィルタリングする。

        Parameters
        ----------
        logits : torch.Tensor
            トークンの予測スコア。形状: (batch_size, vocab_size)
        p : float
            累積確率の閾値。

        Returns
        -------
        torch.Tensor
            フィルタリングされた logits。形状: (batch_size, vocab_size)
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits.scatter_(1, indices_to_remove, float('-inf'))
        return logits
    
    def generate_default(self, input_ids, attention_mask = None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(self.input_seg_len, input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def generate(self, input_ids:torch.Tensor, **generate_kwargs):
        with torch.no_grad():
            if self.is_memory_all is False:
                self.memory_state = None
            elif self.memory_state is not None:
                self.memory_state.detach()  # メモリ状態の勾配を計算しないようにする

            # 入力テンソルをセグメント化してサイズを調整　return: [{'input_ids': 分割1, 'attention_mask': 分割1}, {'input_ids': 分割2, 'attention_mask': 分割2}, ...]
            segmented = self.segment(self.input_seg_len, input_ids=input_ids)

            for seg_num, segment in enumerate(segmented[:-1]):  # 最後のセグメント以外
                # メモリセルに入力テンソルを渡し、出力と新しいメモリ状態を取得
                cell_out, self.memory_state = self.memory_cell(
                    **segment, memory_state=self.memory_state, output_hidden_states=True
                )
                
            curr_segment = segmented[-1]
            """
            outs = []
            for i in range(math.ceil(generate_kwargs["max_length"] / self.input_seg_len)):
                out = self.memory_cell.generate(
                    **curr_segment, 
                    memory_state=self.memory_state, 
                    max_length=min(generate_kwargs["max_length"] - i * self.input_seg_len, self.input_seg_len - curr_segment["input_ids"].shape[-1]), 
                    **generate_kwargs)
                outs.append(out)
                
            for out in outs:
                for key, value in out.items():
                    curr_segment[key] = torch.cat((curr_segment[key], value), dim = -1)
                self.memory_state = out["memory_state"]
            """

            output_ids = None
            if generate_kwargs.get("max_length") is None:
                length = generate_kwargs.get("max_new_tokens", 25)
            else:
                length = generate_kwargs.get("max_length") - curr_segment["input_ids"].shape[-1]

            for ind in range(length):
                # メモリセルに入力テンソルを渡し、出力と新しいメモリ状態を取得
                out, next_memories = self.memory_cell(**curr_segment, memory_state=self.memory_state, output_hidden_states=True)
                logits = out["logits"][:,-1] # (batch_size, vocab_size)
                sampled = self.sample_next_token(logits, temperature = generate_kwargs.get("temperature", 1), top_k = generate_kwargs.get("top_k", 0.9), top_p = generate_kwargs.get("top_p", 0.9), do_sample = generate_kwargs.get("do_sample", False)) # サンプリング (batch_size, 1)
                #filtered_logits = self.top_k(logits, generate_kwargs.get("top_k", 0.9)) # トップkの確率を取得
                #sampled = self.gumbel_sample(filtered_logits, temperature = generate_kwargs.get("temperture", 1)).unsqueeze(1) # サンプリング (batch_size, 1)
                
                output_ids = sampled if output_ids is None else torch.cat((output_ids, sampled), dim = 1)
                
                curr_segment["input_ids"] = torch.cat((curr_segment["input_ids"], sampled), dim = -1) # セグメントにサンプリングされたトークンを追加 (batch_size, seq_len)
                #curr_segment["attention_mask"] = torch.cat((curr_segment["attention_mask"], torch.ones_like(sampled)), dim = -1) # セグメントのアテンションマスクを更新

                if curr_segment["input_ids"].shape[-1] > self.input_seg_len: # セグメントサイズを超えた場合
                    for key, value in curr_segment.items():
                        curr_segment[key] = value[:, -1:] # セグメントサイズに切り詰める
                    self.memory_state = next_memories # メモリ状態を更新

            return output_ids

    def generate_with_tokenizer(self, tokenizer, input_text, **generate_kwargs):
        if isinstance(input_text, str):
            tok = tokenizer(input_text, return_tensors="pt")
            tok["input_ids"] = tok["input_ids"]
            tok["attention_mask"] = tok["attention_mask"]
        else:
            tok = tokenizer(input_text)
            for k, v in tok.items():
                pd = tokenizer.pad_token_id if k != 'attention_mask' else 0
                tok[k] = pad_sequence([torch.tensor(o) for o in v], padding_value=pd, padding_side="left").T
                
        output_ids = self.generate(tok["input_ids"], **generate_kwargs)
        
        if isinstance(input_text, str):
            return tokenizer.decode(torch.cat((tok["input_ids"][0], output_ids[0]), dim=0), skip_special_tokens=True)
        else:
            return tokenizer.batch_decode(torch.cat((tok["input_ids"], output_ids), dim=-1), skip_special_tokens=True)

    def can_generate(self):
        return True

