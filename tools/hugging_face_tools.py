import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import tqdm

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM



class HuggingFaceLlmTools(object):
    
    def __init__(self, llm_name:str, model_dir:str, tokenizer_dir:str, tokenizer_params:Optional[str]=None, system_prompt:Optional[str]=None, device='cpu') -> None:
        self.llm_name = llm_name # e.g., llama31-8b, qwen3-8b, qwen3-14b, ...
        self.device = device
        self.think_id = None
        self.system_prompt = system_prompt
        tokenizer_params = {} if tokenizer_params is None else tokenizer_params
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **tokenizer_params)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
        assert system_prompt is None or 'system' in self.tokenizer.chat_templat

    def tokenize_llm_inputs(self, input_text:str, enable_thinking:bool=True, no_grad:bool=True) -> Dict[str, torch.Tensor]:
        '''Preprocesses and tokenizes input text according to the specified LLM format.
        '''
        ctx = torch.no_grad if no_grad else torch.enable_grad
        with ctx():
            match self.llm_name.split('-')[0]:
                case 'llama31':
                    input_text = '<|start_header_id|>user<|end_header_id|>\n' + input_text + ' <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>'
                    # Encode input text
                    inputs = self.tokenizer(input_text, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                case 'qwen3' | 'qwen3.5':
                    messages = [{"role": "user", "content": input_text}] if self.system_prompt is None else [{'role': 'system', 'content': self.system_prompt}, {"role": "user", "content": input_text}]
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
                    )
                    inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                case _:
                    raise ValueError(f"Unknown LLM: {self.llm_name}")
            return inputs

    def decode_llm_outputs(self, output_sequences:torch.Tensor, input_len:int=0, no_grad:bool=True) -> Tuple[str, Optional[str]]:
        '''Decodes generated token sequences into text, with model-specific postprocessing.
        '''
        ctx = torch.no_grad if no_grad else torch.enable_grad
        with ctx():
            output_ids = output_sequences[0][input_len:].tolist()
            match self.llm_name.split('-')[0]:
                case 'llama31':
                    output_text = (self.tokenizer.decode(output_ids, skip_special_tokens=True), None)
                case 'qwen3' | 'qwen3.5':
                    try:
                        # rindex finding 151668 (</think>)
                        if self.think_id is None:
                            self.think_id = self.tokenizer.convert_tokens_to_ids("</think>")
                        index = len(output_ids) - output_ids[::-1].index(self.think_id)
                    except ValueError:
                        index = 0
                    thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).replace("<think>", "").strip()
                    content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
                    output_text = (content, thinking_content)
                case _:
                    raise ValueError(f"Unknown LLM: {self.llm_name}")
            return output_text

    def get_llm_outputs(self, input_text:str, enable_thinking:bool=True, no_grad:bool=True, output_hidden_states:bool=True, **llm_params) -> Tuple[Tuple[str, Optional[str]], Optional[np.ndarray]]:
        '''Generates LLM outputs and extracts corresponding hidden-state embeddings.
        '''
        ctx = torch.no_grad if no_grad else torch.enable_grad
        with ctx():
            inputs = self.tokenize_llm_inputs(input_text=input_text, enable_thinking=enable_thinking, no_grad=no_grad)                             # Encode input text
            outputs = self.model.generate(**inputs, return_dict_in_generate=True, output_hidden_states=output_hidden_states, **llm_params) # Generate output from the model
            output_text = self.decode_llm_outputs(output_sequences=outputs['sequences'], input_len=len(inputs['input_ids'][0]), no_grad=no_grad)
            if output_hidden_states:
                hidden_states = torch.vstack(outputs.hidden_states[-1]).squeeze().cpu().float().numpy() # shape: [hidden_layer_num, hidden_embedding_dim]; hidden_states[-1] captures the last token
                output_tuple = (output_text, hidden_states)
            else:
                output_tuple = (output_text, None)
            del inputs
            del outputs
            torch.cuda.empty_cache()
            return output_tuple

    def enumerate_llm_outputs(self, queries:Iterator[str], enable_thinking:bool=True, no_grad:bool=True, output_hidden_states:bool=True, **llm_params) -> Tuple[DataFrame, np.ndarray]:
        '''Iteratively generates LLM outputs and embeddings for a set of input units based on a query template.
        '''
        description_llm, hidden_states_llm = [], []
        for input_text in tqdm.tqdm(queries):
            # elem = elem if isinstance(elem, (list, tuple)) else (elem,)
            # input_text = query_template.format(*elem)
            output_text, hidden_states = self.get_llm_outputs(input_text=input_text, enable_thinking=enable_thinking, no_grad=no_grad, output_hidden_states=output_hidden_states, **llm_params)
            description_llm.append([*output_text])
            if output_hidden_states:
                hidden_states_llm.append(hidden_states)
        description_llm = DataFrame(description_llm, columns=[f'Output_{self.llm_name}', f'Thinking_{self.llm_name}'])
        hidden_states_llm = np.array(hidden_states_llm)
        return description_llm, hidden_states_llm
