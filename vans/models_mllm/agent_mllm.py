import os
from peft import get_peft_model
from peft import LoraConfig
import torch.nn.functional as F
import math
from typing import List
import psutil
import torch
from torch import nn
from torchvision import transforms as v2
from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from .qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy
from dataclasses import dataclass

@dataclass
class MLLMSamples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int
    query_embeds: Optional[torch.Tensor] = None
    generated_texts: Optional[List[str]] = None
    video_inputs: Optional[Any] = None
    has_queries: Optional[Any] = None
    max_sequence_length: Optional[Any] = None    
    max_generate_length: Optional[Any] = None

class MLLMAgentConfig(PretrainedConfig):
    def __init__(
        self,
        mllm_id: str = "Qwen2.5-VL",
        base_model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        mllm_lora_config = {
            "r": 8,  
            "lora_alpha": 32,  
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,  
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "train_lora": True,
        },
        num_visionqueries: int = 0,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 256,
        system_prompt: str = "You will be given a video. Your task is to predict the next event based on the input video and the user's instructions. Please begin by providing your detailed reasoning between the [Think][/Think] tags, followed by your description of the next event within the [Ans][/Ans] tags.",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.base_model_name_or_path = base_model_name_or_path
        self.num_visionqueries = num_visionqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.system_prompt = system_prompt
        self.mllm_lora_config = mllm_lora_config


class MLLMAgent(PreTrainedModel):
    config_class = MLLMAgentConfig

    def __init__(
        self,
        config: MLLMAgentConfig = MLLMAgentConfig(),
        mllm_pretrained_path=None,
        trained_patameter_ckpt_path: Optional[str] = None,
    ) -> None:
        
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config
        if mllm_pretrained_path is not None:
            config.base_model_name_or_path = mllm_pretrained_path

        if "Qwen2.5-VL" in config.mllm_id:
            self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.base_model_name_or_path, 
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                use_cache=False,
            )

            self.mllm_backbone.model.config.use_sliding_window = False
            self.mllm_backbone.model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings

            if config.num_visionqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_visionqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_visionqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            if config.num_visionqueries > 0:
                self.mllm_backbone.model.embed_tokens.weight.requires_grad_(True)
            #self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook)

            self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
            # self.mllm_backbone.lm_head = nn.Identity()

            self.tokenizer = AutoProcessor.from_pretrained(config.base_model_name_or_path)
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None

        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}")
        

        if config.mllm_lora_config is not None:
            print("############# ACTIVATE LORA TRAINING FOR MLLM #############")
            mllm_lora_config = LoraConfig(
                        task_type=config.mllm_lora_config['task_type'],
                        target_modules=config.mllm_lora_config['target_modules'],  
                        #inference_mode=config.mllm_lora_config['train_lora'],
                        r=config.mllm_lora_config['r'], 
                        lora_alpha=config.mllm_lora_config['lora_alpha'], 
                        lora_dropout=config.mllm_lora_config['lora_dropout'],
                        bias=config.mllm_lora_config['bias'],  
            )
            self.mllm_backbone = get_peft_model(self.mllm_backbone, mllm_lora_config)

            if not config.mllm_lora_config.get('train_lora', True):
                for param in self.mllm_backbone.parameters():
                    param.requires_grad = False

            if config.num_visionqueries > 0:
                self.mllm_backbone.model.model.embed_tokens.weight.requires_grad_(True)
                self.mllm_backbone.base_model.model.model.embed_tokens.weight.register_hook(freeze_hook)
            self.mllm_backbone.print_trainable_parameters()

            # load trained parameters for mllm
            if trained_patameter_ckpt_path is not None and os.path.exists(trained_patameter_ckpt_path):
                if trained_patameter_ckpt_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    try:
                        custom_state_dict = load_file(trained_patameter_ckpt_path, device="cpu")
                    except Exception as e:
                        custom_state_dict = {}
                else:
                    custom_state_dict = torch.load(trained_patameter_ckpt_path, map_location="cpu", weights_only=False)
                
                if custom_state_dict:
                    model_state_dict = self.mllm_backbone.state_dict()
                    updated_state_dict = {}

                    for name, param in custom_state_dict.items():
                        updated_name = name.replace('mllmagent.mllm_backbone.', '').replace('pipe.mllm.mllm_backbone.','')
                        if updated_name in model_state_dict:
                            if param.shape == model_state_dict[updated_name].shape:
                                updated_state_dict[updated_name] = param
                    
                    if updated_state_dict:
                        model_state_dict.update(updated_state_dict)
                        self.mllm_backbone.load_state_dict(model_state_dict)
                        
                        del model_state_dict, updated_state_dict
                    
    
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_visionqueries = config.num_visionqueries
        self.tokenizer.system_prompt = config.system_prompt
        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id
        
        if config.num_visionqueries > 0:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<pad_token_{i}>"
                        for i in range(num_embeddings - len(tokenizer))
                    ]
                }
            )
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<BOQ>", "<EOQ>"]
                    + [f"<Query{i}>" for i in range(self.tokenizer.num_visionqueries)]
                }
            )
            self.boq_token_id = tokenizer.convert_tokens_to_ids("<BOQ>")
            self.eoq_token_id = tokenizer.convert_tokens_to_ids("<EOQ>")

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.model.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
            except:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
                pass

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    def get_resize_fn(self):
        return self.resize_fn

    @staticmethod
    @torch.no_grad()
    def tokenize(
        tokenizer, 
        preprocessed_qwen_emb=None,
        is_train=False, 
        caption=None, 
        instructions=None, 
        thinking=None,
        add_query=False, 
        video=None, 
        text_response=None, 
        add_generation_prompt=True, 
        device='cuda:1', 
        torch_dtype=torch.bfloat16,
    ):

        if not isinstance(caption, List):
            caption = [caption]
        if not isinstance(instructions, List):
            instructions = [instructions]
        if not isinstance(thinking, List):
            thinking = [thinking]

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                        [
                            {
                                "type": "text", 
                                "text": tokenizer.system_prompt
                            }
                        ]
                    )
                }
            ]
            if tokenizer.system_prompt is not None
            else []
        )
        

        if not add_generation_prompt or tokenizer.num_visionqueries <= 0:
            suffix = ""
        else:
            suffix = (
                "\n<|im_end|><BOQ>"
                + "".join([f"<Query{i}>" for i in range(tokenizer.num_visionqueries)])
                + "<EOQ><|im_end|>"
            )


        if not isinstance(video, list):
            video = [video]

        messages_batch = [
            prefix
            + 
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": vids},
                        {"type": "text", "text": ins},
                    ],
                },
            ]
            for vids, ins in zip(video, instructions)
        ]

        texts_processed = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in messages_batch
        ]

        video_inputs_batch = []
        for msg in messages_batch:
            image_inputs, video_inputs = process_vision_info(msg, device)
            video_inputs_batch.append(video_inputs)

        if add_query:
            if tokenizer.num_visionqueries > 0:
                endtoken=''
            else:
                endtoken = '<|im_end|>'

            if thinking is not None:
                try:
                    texts_processed = [texts_processed[i] + "[Think]" + thinking[i] + "[/Think]" + "[Ans]" + caption[i] + f"[/Ans]{endtoken}" for i in range(len(texts_processed))]
                except:
                    print("WTONG TEXT",thinking,caption)
                    texts_processed = [texts_processed[i] + f"{endtoken}" for i in range(len(texts_processed))]
            else:
                texts_processed = [texts_processed[i] + "[Ans]" + caption[i] + f"[/Ans]{endtoken}" for i in range(len(texts_processed))]


        if add_query and tokenizer.num_visionqueries > 0:
            texts_processed = [p + suffix for p in texts_processed]
        
        if is_train:
            ori_texts_processed = ori_video_inputs_batch = None
        else:
            ori_texts_processed = deepcopy(texts_processed)
            ori_video_inputs_batch = deepcopy(video_inputs_batch)

        inputs = tokenizer(
            text=texts_processed,
            videos=video_inputs_batch,
            return_tensors="pt",
            padding=True,
            padding_side='left',
        )

        if add_query:
            ASSISTANT_TOKEN = tokenizer(text='assistant')['input_ids'][0][0]
            answer_start_idx = torch.where(inputs['input_ids'] == ASSISTANT_TOKEN)[1]
            inputs['labels'] = torch.full_like(inputs['attention_mask'], fill_value=-100)

            if tokenizer.num_visionqueries > 0:
                BOQ_TOKEN = tokenizer(text='<BOQ>')['input_ids'][0][0]
                EOQ_TOKEN = tokenizer(text='<EOQ>')['input_ids'][0][0]
                answer_end_idx = torch.where(inputs['input_ids'] == BOQ_TOKEN)[1]
                query_end_idx = torch.where(inputs['input_ids'] == EOQ_TOKEN)[1]

                for batch_idx in range(inputs['input_ids'].shape[0]):
                    start_idx = answer_start_idx[batch_idx]
                    end_idx = answer_end_idx[batch_idx]
                    query_idx = query_end_idx[batch_idx]

                    inputs['attention_mask'][batch_idx, end_idx+1: query_idx] = -1
                    inputs['labels'][batch_idx, start_idx + 1 :end_idx] = inputs['input_ids'][batch_idx, start_idx + 1 :end_idx]
                    assert inputs['labels'][batch_idx][end_idx-1] == inputs['input_ids'][batch_idx][end_idx-1] == torch.tensor(tokenizer.tokenizer.eos_token_id)
            else:
                for batch_idx in range(inputs['input_ids'].shape[0]):
                    start_idx = answer_start_idx[batch_idx]
                    inputs['labels'][batch_idx, start_idx + 1 :] = inputs['input_ids'][batch_idx, start_idx + 1 :]

        inputs = inputs.to(device)
        return inputs, ori_video_inputs_batch, ori_texts_processed

    def extract_query_output(self, mllm_input, output_lm):

        prompt_embeds = output_lm.hidden_states[-1]
        input_ids = mllm_input['input_ids']
        boi_pos = torch.where(input_ids == self.boq_token_id)[1]
        eoi_pos = torch.where(input_ids == self.eoq_token_id)[1]

        # Create mask for selecting tokens between BOI and EOI
        batch_size, seq_len = input_ids.shape
        indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(batch_size, -1)
        mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])

        query_embeds = prompt_embeds[mask].view(
            batch_size, -1, prompt_embeds.size(-1)
        )
        query_attention_mask = mllm_input['attention_mask'][mask].view(batch_size, -1)  
        return query_embeds, query_attention_mask

    def forward(self, batch):
        

        captions = [batch.get("ENG_GT_Caption", None)] # List of Caption
        video_paths = [batch.get("input_video_path", None)] # List of Video Paths
        instructions = [batch.get("ENG_Instruction", None)]
        thinking = [batch.get("ENG_Think", None)]
        tokenizer = self.tokenizer

        with torch.no_grad():
            mllm_input, _, _ = self.tokenize(
                tokenizer,
                preprocessed_qwen_emb=None, 
                is_train=True,
                add_query=True,
                instructions=instructions,
                thinking=thinking,
                caption=captions, 
                video=video_paths, 
                device=self.mllm_backbone.device, 
                torch_dtype=self.mllm_backbone.dtype,
            )

        output_lm = self.mllm_backbone(
            **mllm_input,
            output_hidden_states=True,
            return_dict=True
        )

        LM_loss = output_lm['loss'] 

        if self.tokenizer.num_visionqueries > 0:
            query_embeds, query_attention_mask = self.extract_query_output(mllm_input, output_lm)
        else:
            query_embeds = output_lm.hidden_states[-1]
            query_attention_mask = None 

        return LM_loss, query_embeds, query_attention_mask
    
    @torch.no_grad() 
    def generate_with_query(self, instructions=None, prompt=None, video_paths=None):
        
        tokenizer = self.tokenizer

        mllm_input_generate_cation, video_inputs_batch, texts_processed = self.tokenize(
            tokenizer, 
            is_train=False,
            add_query=False,
            instructions=instructions,
            caption=prompt, 
            video=video_paths, 
            device=self.mllm_backbone.device, 
            torch_dtype=self.mllm_backbone.dtype,
        )

        if tokenizer.num_visionqueries == 0:
            mllm_output = self.mllm_backbone.generate(
                **mllm_input_generate_cation,
                max_new_tokens=1024,
                do_sample=True, 
                temperature=0.7,  
                top_p=0.9,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            generated_ids = mllm_output.sequences
            query_embeds = mllm_output.hidden_states[-1][-1]

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(mllm_input_generate_cation.input_ids, generated_ids)]
            output_texts = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            print("MLLM Generate Response: ", output_texts)

        else:
            generated_ids = self.mllm_backbone.generate(
                **mllm_input_generate_cation,
                max_new_tokens=256,
                do_sample=True, 
                temperature=0.7,  
                top_p=0.9,
            )

            generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(mllm_input_generate_cation.input_ids, generated_ids)]

            output_texts = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            print("MLLM Generate Response: ", output_texts)
            
            suffix = ("<|im_end|><BOQ>" + "".join([f"<Query{i}>" for i in range(tokenizer.num_visionqueries)]) + "<EOQ><|im_end|>")
            query_input_texts = [tp + ot + suffix for tp, ot in zip(texts_processed, output_texts)]
            
            # get input_ids with generated caption and query
            mllm_input_generate_query = tokenizer(
                text=query_input_texts,
                videos=video_inputs_batch,
                return_tensors="pt",
                padding=True,
                padding_side='left',
            ).to(self.mllm_backbone.device) 

            BOQ_TOKEN = tokenizer(text='<BOQ>')['input_ids'][0][0]
            EOQ_TOKEN = tokenizer(text='<EOQ>')['input_ids'][0][0]
            answer_end_idx = torch.where(mllm_input_generate_query['input_ids'] == BOQ_TOKEN)[1]
            query_end_idx = torch.where(mllm_input_generate_query['input_ids'] == EOQ_TOKEN)[1]
            mllm_input_generate_query['attention_mask'][:, answer_end_idx+1: query_end_idx] = -1

            mllm_input_generate_query = mllm_input_generate_query.to(self.mllm_backbone.device)

            output_lm = self.mllm_backbone(
                **mllm_input_generate_query,
                output_hidden_states=True,
                return_dict=True
            )
            
            if self.tokenizer.num_visionqueries > 0:
                query_embeds, query_attention_mask = self.extract_query_output(mllm_input_generate_query, output_lm)

        return output_texts, query_embeds

    @torch.no_grad()
    def generate_grpo_samples(self, batch=None):
        samples_list = []
        
        video_paths = batch.get("input_video_path", [None])
        instructions = batch.get("ENG_Instruction", [""])
        
        tokenizer = self.tokenizer
        
        max_prompt_length = 512  
        max_generate_length = 256
        max_query_length = 0
        
        max_sequence_length = max_prompt_length + max_generate_length + max_query_length
        
        for instruction, video_path in zip(instructions, video_paths):

            repeated_instructions = [instruction] * 1
            repeated_video_paths = [video_path] * 1
            
            mllm_input_generate_cation, video_inputs_batch, texts_processed = self.tokenize(
                tokenizer, 
                is_train=False,
                add_query=False,
                instructions=repeated_instructions,
                caption=None, 
                video=repeated_video_paths, 
                device=self.mllm_backbone.device, 
                torch_dtype=self.mllm_backbone.dtype,
            )
            
            all_complete_sequences = []  
            all_output_texts = []
            all_query_embeds = []
            
            if tokenizer.num_visionqueries == 0:

                mllm_output = self.mllm_backbone.generate(
                    **mllm_input_generate_cation,
                    max_new_tokens=max_generate_length,
                    do_sample=True, 
                    temperature=0.7,  
                    top_p=0.9,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                
                generated_ids = mllm_output.sequences
                all_complete_sequences.append(generated_ids)
                
                if hasattr(mllm_output, 'hidden_states') and mllm_output.hidden_states:
                    query_embeds = mllm_output.hidden_states[-1][-1]
                    all_query_embeds.append(query_embeds)
                else:
                    with torch.no_grad():
                        outputs = self.mllm_backbone(
                            input_ids=generated_ids,
                            attention_mask=(generated_ids != tokenizer.tokenizer.pad_token_id),
                            output_hidden_states=True
                        )
                        query_embeds = outputs.hidden_states[-1][:, -1:]
                        all_query_embeds.append(query_embeds)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(mllm_input_generate_cation.input_ids, generated_ids)
                ]
                output_texts = tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                all_output_texts.extend(output_texts)

                print(f"MLLM Generate Response: {output_texts}")

            else:

                mllm_output = self.mllm_backbone.generate(
                    **mllm_input_generate_cation,
                    max_new_tokens=max_generate_length,
                    do_sample=True, 
                    temperature=0.7,  
                    top_p=0.9,
                    return_dict_in_generate=True,
                )
                
                generated_ids_step1 = mllm_output.sequences
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(mllm_input_generate_cation.input_ids, generated_ids_step1)
                ]
                output_texts = tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                all_output_texts.extend(output_texts)
                print(f"MLLM Generate Response: {output_texts}")


                suffix = ("<|im_end|><BOQ>" + "".join([f"<Query{i}>" for i in range(tokenizer.num_visionqueries)]) + "<EOQ><|im_end|>")
                query_input_texts = [tp + ot + suffix for tp, ot in zip(texts_processed, output_texts)]
                
                mllm_input_generate_query = tokenizer(
                    text=query_input_texts,
                    videos=video_inputs_batch,
                    return_tensors="pt",
                    padding=True,
                    padding_side='left',
                    max_length=max_sequence_length,
                    truncation=True
                ).to(self.mllm_backbone.device)

                BOQ_TOKEN = tokenizer(text='<BOQ>')['input_ids'][0][0]
                EOQ_TOKEN = tokenizer(text='<EOQ>')['input_ids'][0][0]
                answer_end_idx = torch.where(mllm_input_generate_query['input_ids'] == BOQ_TOKEN)[1]
                query_end_idx = torch.where(mllm_input_generate_query['input_ids'] == EOQ_TOKEN)[1]
                mllm_input_generate_query['attention_mask'][:, answer_end_idx+1: query_end_idx] = -1

                mllm_input_generate_query = mllm_input_generate_query.to(self.mllm_backbone.device)


                output_lm = self.mllm_backbone(
                    **mllm_input_generate_query,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                if self.tokenizer.num_visionqueries > 0:
                    query_embeds, query_attention_mask = self.extract_query_output(mllm_input_generate_query, output_lm)
                    all_query_embeds.append(query_embeds)
                
                all_complete_sequences.append(mllm_input_generate_query.input_ids)

            if all_complete_sequences:
                complete_sequences_tensor = torch.cat(all_complete_sequences, dim=0)
                
                if complete_sequences_tensor.size(1) > max_sequence_length:
                    complete_sequences_tensor = complete_sequences_tensor[:, :max_sequence_length]
                    print(f"Warning: Sequence truncated to {max_sequence_length} tokens")
                elif complete_sequences_tensor.size(1) < max_sequence_length:
                    padding = torch.full(
                        (complete_sequences_tensor.size(0), max_sequence_length - complete_sequences_tensor.size(1)),
                        tokenizer.tokenizer.pad_token_id,
                        device=complete_sequences_tensor.device,
                        dtype=complete_sequences_tensor.dtype
                    )
                    complete_sequences_tensor = torch.cat([complete_sequences_tensor, padding], dim=1)
                
                attention_mask = (complete_sequences_tensor != tokenizer.tokenizer.pad_token_id).to(dtype=torch.long)
                
                if tokenizer.num_visionqueries == 0:
                    prompt_length = mllm_input_generate_cation.input_ids.size(1)
                    prompt_length = min(prompt_length, max_sequence_length - max_generate_length)
                    
                    response_ids = complete_sequences_tensor[:, prompt_length:prompt_length + max_generate_length]
                    
                    if response_ids.size(1) < max_generate_length:
                        padding = torch.full(
                            (response_ids.size(0), max_generate_length - response_ids.size(1)),
                            tokenizer.tokenizer.pad_token_id,
                            device=response_ids.device,
                            dtype=response_ids.dtype
                        )
                        response_ids = torch.cat([response_ids, padding], dim=1)
                    
                    action_mask = (response_ids != tokenizer.tokenizer.eos_token_id) & (response_ids != tokenizer.tokenizer.pad_token_id)
                    num_actions = max_generate_length
                    
                else:

                    prompt_length = mllm_input_generate_cation.input_ids.size(1)
                    total_response_length = max_generate_length + max_query_length
                    
                    response_start = min(prompt_length, max_sequence_length - total_response_length)
                    response_ids = complete_sequences_tensor[:, response_start:response_start + total_response_length]
                    
                    if response_ids.size(1) < total_response_length:
                        padding = torch.full(
                            (response_ids.size(0), total_response_length - response_ids.size(1)),
                            tokenizer.tokenizer.pad_token_id,
                            device=response_ids.device,
                            dtype=response_ids.dtype
                        )
                        response_ids = torch.cat([response_ids, padding], dim=1)
                    
                    action_mask = (response_ids != tokenizer.tokenizer.eos_token_id) & (response_ids != tokenizer.tokenizer.pad_token_id)
                    num_actions = total_response_length
                
                action_mask = action_mask.to(dtype=torch.long)
                
                actual_response_length = action_mask.float().sum(dim=-1)
                
                samples = MLLMSamples(
                    prompt_response_ids=complete_sequences_tensor,  
                    response_ids=response_ids,                      
                    prompt=instruction,                             
                    answer=all_output_texts,                        
                    attention_mask=attention_mask,                  
                    action_mask=action_mask,                        
                    num_actions=num_actions,                        
                    response_length=actual_response_length,         
                    query_embeds=torch.cat(all_query_embeds, dim=0) if all_query_embeds else None,
                    generated_texts=all_output_texts,
                    video_inputs=video_inputs_batch,
                    has_queries=(tokenizer.num_visionqueries > 0),
                    max_sequence_length=max_sequence_length,
                    max_generate_length=max_generate_length,
                )
                samples_list.append(samples)
            
        return samples_list


    def get_action_log_probs(self, input_ids, attention_mask, num_actions):

        outputs = self.mllm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        
        log_probs_labels = log_probs.gather(
            dim=-1, 
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        prompt_length = input_ids.size(1) - num_actions
        action_log_probs = log_probs_labels[:, prompt_length-1:prompt_length+num_actions-1]
        
        return action_log_probs


    def compute_grpo_loss(self, inputs):
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(prompt_response_ids, attention_mask, num_actions)
        
        beta = inputs.get('beta', 0.0)
        
        if beta != 0.0:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        
        old_action_log_probs = action_log_probs.detach() # inputs['old_action_log_probs'] if hasattr(self.args, 'num_iterations') and self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - 0.2, 1 +  0.2) # clip_eps = 0.2
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        
        if beta != 0.0:
            per_token_loss = per_token_loss + beta * k3
        
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1).clamp(min=1)
        loss = loss.mean()
        
        return loss