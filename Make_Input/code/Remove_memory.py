from Make_Input import Make_input
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import json
from tqdm import tqdm

def main():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memories_dict = {}
    
    abs_path = os.path.dirname(__file__)
    input_text = "あ"
    json_file_path = f"../result/memories_used/{input_text}_memories/input_{input_text}.json"
    
    result_path = "../result/remove_memory"
    txt_file_name = 'remove_memory.txt'

    with open(os.path.join(abs_path,json_file_path)) as f:
        memories_dict = json.load(f)
    blocks = [2,4,18,20]
    
    with torch.no_grad():
        now_idx = 0
        for arr in memories_dict[input_text]:
            stage = 0
            for i in range(4):
                if now_idx < blocks[i]:
                    stage = i
                    break
            block = now_idx-blocks[stage-1] if not stage == 0 else now_idx
            for idx in arr:
                model.encoder.encoder.layers[stage].blocks[block].intermediate.dense.weight[idx] = 0.0
                print(stage,block,idx)
            now_idx+=1

    model.to(device)
    
    #path
    path = os.path.join(abs_path,result_path)
    txt_file_path = os.path.join(path,txt_file_name)
    
    make_input = Make_input(text=input_text,font_size=40,text_pos=(0,256))
    img, _ = make_input.create_image()
    task_prompt = "<s_iitcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(img, return_tensors="pt").pixel_values
    model.eval()
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


def make_acc_json():
    abs_path = os.path.dirname(__file__)
    result_path = '../result/remove_memory'
    path = os.path.join(abs_path,result_path)
    json_file_name = 'remove_result.json'
    pos_dict = {}
    BG_H = 2560
    BG_W = 1920
    hs = [i*BG_H//10 for i in range(10)]
    ws = [i*BG_W//10 for i in range(10)]
    ps = []
    for h in hs:
        for w in ws:
            ps.append((w,h))
    for pos in ps:
        pos_dict[str(pos)] = 0
    with open(os.path.join(path,json_file_name),'r') as f:
        result_dict = json.load(f)
        for word,dict in result_dict.items():
            for pos, ret in dict.items():
                if ret:
                    pos_dict[pos] += 1
    name = 'remove_acc.json'
    with open(os.path.join(path,name),'w') as f:
        json.dump(pos_dict,f,indent=2)

def get_cmap_by_acc():
    result_path = '../result/remove_momory'
    json_file_name = "remove_acc.json"
    result_path = os.path.join(result_path,json_file_name)
    abs_path = os.path.dirname(__file__)
    path = os.path.join(abs_path,result_path)
    acc_dict = {}
    with open(path,'r') as f:
        acc_dict = json.load(f)
    
    ex = r"\((\d+),\s*(\d+)\)"
    cmap_np = np.zeros((2560,1920))
    for pos,cnt in acc_dict.items():
        ma = re.match(ex,pos)
        if ma:
            x = int(ma.group(1))
            y = int(ma.group(2))
            cmap_np[y:y+2560//10,x:x+1920//10] = cnt/62
    
    save_path = '../result/position'
    fig_name = 'cmap_acc'
    plt.figure(figsize=(10,10))
    plt.imshow(cmap_np,cmap='viridis')
    plt.title("accuracy depending on the position of the word")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.colorbar(label='accuracy')
    plt.savefig(os.path.join(save_path,fig_name))

if __name__ == '__main__':
    main()
    #make_acc_json()
    #get_cmap_by_acc()
