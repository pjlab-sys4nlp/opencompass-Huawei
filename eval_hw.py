import argparse
import os
import os.path as osp
import re
import sys
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config
from sentencepiece import SentencePieceProcessor

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config as InternLMConfig
from internlm.initialize import initialize_distributed_env
from internlm.utils.storage_manager import init_storage_manager
from internlm.utils.storage_manager import get_fns, llm_load

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.tasks.openicl_infer import OpenICLInferTask
from opencompass.registry import MODELS

sys.path.append('.')

from hw_eval.generation_tools import LLMGenerator

basic_config = dict(num_chunks=1,
                    checkpoint=False,
                    dtype=torch.half,
                    embed_split_hidden=False,
                    num_layers=40,
                    hidden_size=5120,
                    vocab_size=150494,
                    embed_grad_scale=1,
                    parallel_output=False,
                    num_attention_heads=40,
                    mlp_ratio=8 / 3,
                    apply_post_layer_norm=False,
                    residual_in_fp32=False,
                    norm_type='rmsnorm',
                    drop_rate=0,
                    attn_drop_rate=0)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


backup = {}


def proxy_off():
    global backup
    if 'http_proxy' in os.environ:
        backup['http_proxy'] = os.environ['http_proxy']
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        backup['https_proxy'] = os.environ['https_proxy']
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        backup['HTTP_PROXY'] = os.environ['HTTP_PROXY']
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        backup['HTTPS_PROXY'] = os.environ['HTTPS_PROXY']
        del os.environ['HTTPS_PROXY']


def proxy_on():
    global backup
    if 'http_proxy' in backup:
        os.environ['http_proxy'] = backup['http_proxy']
    if 'https_proxy' in backup:
        os.environ['https_proxy'] = backup['https_proxy']
    if 'HTTP_PROXY' in backup:
        os.environ['HTTP_PROXY'] = backup['HTTP_PROXY']
    if 'HTTPS_PROXY' in backup:
        os.environ['HTTPS_PROXY'] = backup['HTTPS_PROXY']


def get_master_node():
    import subprocess
    if os.getenv('SLURM_JOB_ID') is None:
        raise RuntimeError('get_master_node can only used in Slurm launch!')
    result = subprocess.check_output(
        'scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1',
        shell=True)
    result = result.decode('utf8').strip()
    return result


class WarpModel(torch.nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class LLMTokenizer(object):

    def __init__(self, tokenizer, max_seq_len=2048, tokenizer_type='llama'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenizer_type = tokenizer_type
        if self.tokenizer_type == 'v4':
            self.bos_token_id = self.pad_token_id = 0
            self.eos_token_id = 1
        elif self.tokenizer_type in ['llama', 'v7']:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 2
        else:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 0

        # This is a hack to fit in with LLama type model
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id
        self.pad_id = self.pad_token_id

    def __call__(self,
                 prompts,
                 padding=True,
                 right_align=False,
                 return_tensors='pt',
                 truncation=True):
        if self.tokenizer_type == 'v4':
            tokens = [[0] + self.encode(x, False, False) for x in prompts]
        elif self.tokenizer_type in ['llama', 'v7']:
            tokens = [[1] + self.encode(x, False, False) for x in prompts]
        else:
            tokens = [self.encode(x, False, False) for x in prompts]

        if truncation:
            tokens = [i[:self.max_seq_len] for i in tokens]

        if padding:
            max_len = max([len(i) for i in tokens])
            if right_align:
                tokens = torch.LongTensor([[self.pad_token_id] *
                                           (max_len - len(i)) + i
                                           for i in tokens])
            else:
                tokens = torch.LongTensor([
                    i + [self.pad_token_id] * (max_len - len(i))
                    for i in tokens
                ])
        return {
            'tokens': tokens.cuda() if torch.cuda.is_available() else tokens
        }

    def encode(self, s: str, bos: bool, eos: bool):
        assert isinstance(s, str)
        s = self._process_meta_tokens(s)
        t = self._tokenize_list_str(s)
        if bos:
            t = [self.bos_token_id] + t
        if eos:
            t = t + [self.eos_token_id]
        return t

    def _process_meta_tokens(self, input_string: str) -> List[Union[str, int]]:
        # Create a pattern to match the META_TOKEN_{NUM} substrings
        pattern = re.compile(r'<META_TOKEN_(\d+)>')

        # Split the input string using the META_TOKEN_{NUM} substrings
        parts = pattern.split(input_string)

        # Combine the parts and tokens in the correct order
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text parts
                if part != '':
                    result.append(part)
            else:  # Meta token parts
                result.append(int(part))

        return result

    def _tokenize_list_str(self, s: Union[str, list]) -> List[int]:
        if isinstance(s, str):
            s = [s]
        assert isinstance(s, list)
        t = []
        for item in s:
            if isinstance(item, str):
                t += self.tokenizer.encode(item)
            elif isinstance(item, int):
                t.append(item)
            else:
                raise ValueError(f'Unsupported type {type(item)}')
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


def load_llm(checkpoint,
             max_seq_len=2048,
             tokenizer_path: Optional[str] = None,
             tokenizer_type: Optional[str] = None,
             module=None,
             model_config_path=None):
    """load_llm函数只支持加载 internlm 格式的模型权重.

    Args:
        checkpoint (_type_): _description_
        max_seq_len (int, optional): _description_. Defaults to 2048.
        tokenizer_path (Optional[str], optional): _description_. Defaults to None.
        tokenizer_type (Optional[str], optional): _description_. Defaults to None.
        module (_type_, optional): _description_. Defaults to None.
        model_config_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    assert tokenizer_path is not None
    ckpts = checkpoint.split(';')
    ckpt = str(ckpts[0])
    tp_local_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    tp_world_size = gpc.get_world_size(ParallelMode.TENSOR)

    # parameter spliting:
    try:
        model_name, cur_iter = osp.realpath(ckpt).split('/')[-2:]
    except Exception as e:
        print(f'{e}，we only support ckpt path format like XX/50/', flush=True)
    else:
        print(f"Found internlm ckpet: {model_name}/{cur_iter}", flush=True)

    # 'model_tp0_pp*.pt' ~ 'model_tp7_pp*.pt'
    save_tp = 0
    for file in os.listdir(ckpt):
        if file.startswith('model_tp') and file.endswith('.pt'):
            save_tp = max(save_tp, int(file[8:].split('_')[0]))
    save_tp += 1
    assert tp_world_size == save_tp, f"tp size is not match!"

    # print args info
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    if tp_rank == 0:
        print(f'Args: ckpt={checkpoint}')

    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    tokenizer = LLMTokenizer(tokenizer,
                             max_seq_len=max_seq_len,
                             tokenizer_type=tokenizer_type)

    config_file = os.path.join(ckpt, 'model_config.pt')
    with open(config_file, 'rb') as f:
        update_config = torch.load(f)

    def convert2run(model_config):
        model_config['dtype'] = torch.half if str(
            model_config['dtype']) == 'torch.float16' else torch.bfloat16
        model_config['parallel_output'] = False
        return model_config

    model_config = basic_config
    model_config.update(update_config)
    model_config = convert2run(model_config)
    model_config.pop('use_swiglu', 'None')
    model_config.pop('init_std', 'None')
    model_config.pop('init_method', 'None')
    model_module = module
    if tokenizer_type in ['llama', 'v6', 'v4']:
        model_config['embed_split_hidden'] = True

    if 'layer_norm_epsilon' in model_config:
        del model_config['layer_norm_epsilon']

    model = model_module(**model_config)
    states = llm_load(os.path.join(ckpt, f'model_tp{tp_local_rank}_pp{0}.pt'), map_location='cpu')
    model = WarpModel(model)  # 因为我们加载的 ckpt 中被amp包了一层，所以要加上 model 前缀
    model.load_state_dict(states, strict=True)
    model = model.half().eval().cuda()
    torch.distributed.barrier()

    generator = LLMGenerator(model, tokenizer, False, forward_kwargs={})

    return model, tokenizer, generator, tp_rank


class InternLM(BaseModel):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 tokenizer_path: Optional[str] = None,
                 model_config: Optional[str] = None,
                 tokenizer_type: Optional[str] = 'v7',
                 meta_template: Optional[Dict] = None):
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path,
                                 tokenizer_type=tokenizer_type,
                                 max_seq_len=max_seq_len)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             tokenizer_path=tokenizer_path,
                             tokenizer_type=tokenizer_type,
                             model_config=model_config)
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None,
                    model_config: Optional[str] = None):
        raise NotImplementedError

    def _load_tokenizer(self, tokenizer_path: str, tokenizer_type: str,
                        max_seq_len: int):
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        tokenizer = LLMTokenizer(tokenizer,
                                 max_seq_len=max_seq_len,
                                 tokenizer_type=tokenizer_type)
        self.tokenizer = tokenizer

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        tokens = self.tokenizer([prompt], truncation=False)['tokens']
        return len(tokens[0])

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        return self.generator.generate(inputs,
                                       generation_kwargs={
                                           'max_gen_len': max_out_len,
                                           'eos_token_id': self.eos_token_id
                                       })

    def get_ppl(self,
                input_texts: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            input_texts (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out.

        Returns:
            List[float]: A list of perplexity scores.
        """
        outputs, inputs = self.generator.get_logits(input_texts)

        shift_logits = outputs[..., :-1, :].contiguous().float()
        shift_labels = inputs['tokens'][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss

@MODELS.register_module()
class HW_InternLM_7B(InternLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None,
                    model_config: Optional[str] = None):

        from internlm.model.modeling_internlm import build_model_with_cfg

        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
            module=build_model_with_cfg,
            model_config_path=model_config)

@MODELS.register_module()
class HW_llama2_70B(InternLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None,
                    model_config: Optional[str] = None):

        from internlm.model.modeling_llama import build_model_with_cfg

        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
            module=build_model_with_cfg,
            model_config_path=model_config)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg['datasets'] = [cfg['datasets']]
    proxy_off()
    init_storage_manager(enable_save_ckpt=False,
                         async_upload=False,
                         async_upload_tmp_folder=None)
    
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12349'
    os.environ['NCCL_SOCKET_IFNAME'] = 'bond0'
    # os.environ['NCCL_IB_HCA'] = 'mlx5_2,mlx5_3,mlx5_4,mlx5_5'

    # 仅仅是为了初始化进程组
    gpc_init_config = InternLMConfig(
        dict(
            parallel=dict(zero1=dict(size=-1, fsdp=False),
                          pipeline=dict(size=1, interleaved_overlap=False),
                          sequence_parallel=False,
                          tensor=world_size),
            model_type='',
            adam=dict(),
            data=dict(),
            model=dict(dtype=torch.half),
            resume_tb_folder='',
            tensorboard_folder='',
            alert_address=None,
            monitor=dict(alert=dict(enable_feishu_alert=False,
                                    feishu_alert_address=None,
                                    light_monitor_address=None)),
        ))

    assert not dist.is_initialized()
    print(os.environ['MASTER_ADDR'], flush=True)
    print(gpc_init_config, flush=True)
    initialize_distributed_env(config=gpc_init_config,
                               launcher='torch',
                               master_port=12349,
                               args_check=False)
    print(f'initialize_distributed_env done', flush=True)

    start_time = time.time()
    inferencer = OpenICLInferTask(cfg)
    inferencer.run()
    end_time = time.time()
    print(f'Infer time elapsed: {end_time - start_time:.2f}s', flush=True)

    start_time = time.time()
    inferencer = OpenICLEvalTask(cfg)
    inferencer.dump_details = True
    inferencer.run()
    end_time = time.time()
    print(f'eval time elapsed: {end_time - start_time:.2f}s', flush=True)
    exit(0)
