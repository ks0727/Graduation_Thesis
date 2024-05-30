from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from Make_Input import Make_input
import os
import re

#function to convert torch.Tensor() to ndarray
def to_np(x:torch.Tensor):
    x = x.to('cpu').detach().numpy().copy()
    return x

def print_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")


def main()->int:
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    #task_prompt = "<s_iitcdip>"
    save = []

    def hook(model,input,output):
        #output_cpu = output.data.cpu()
        #output_cpu = output_cpu.squeeze()
        save.append(output.detach().cpu().squeeze())
    
    for i in range(4):
        jmax = 14 if i == 2 else 2
        for j in range(jmax):
            pass
            model.encoder.encoder.layers[i].blocks[j].intermediate.register_forward_hook(hook)
            
    datasets = load_dataset("hf-internal-testing/example-documents", split="test")
    image = datasets["image"][0]
    
    abs_path = os.path.dirname(__file__)
    result_path = "../result"
    path = os.path.join(abs_path,result_path)
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    out = model.encoder(pixel_values.to(device))
    del out
    torch.cuda.empty_cache()
    mx_memories = []
    for si in range(len(save)):
        output_i = save[si]
        memories = output_i.sum(dim=0)
        mx_idx = torch.argsort(-memories)
        mx_idx = mx_idx.data.cpu()[:11].tolist()
        mx_memories.append(mx_idx)
        mx_idx = 0
    save = []
    blocks = [2,4,18,20]
    
    with torch.no_grad():
        now_idx = 0
        for arr in mx_memories:
            stage = 0
            for i in range(4):
                if now_idx < blocks[i]:
                    stage = i
                    break
            block = now_idx-blocks[stage-1] if not stage == 0 else now_idx
            now = 0
            for idx in arr:
                model.encoder.encoder.layers[stage].blocks[block].intermediate.dense.weight[idx] = 0.0
            now_idx+=1
    
    task_prompt = "<s_iitcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(sequence)


if __name__ == '__main__':
    main()