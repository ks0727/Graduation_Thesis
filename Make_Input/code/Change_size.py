from Make_Input import Make_input
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import sys

def main():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    #making list of alphabets and numbers  print(cnt)
    lower = [chr(ord('a')+i) for i in range(26)] #a~z
    upper = [chr(ord('A')+i) for i in range(26)] #A~Z
    num = [chr(ord('0')+i) for i in range(10)] #0~9
    words = lower+upper+num
    #path
    abs_path = os.path.dirname(__file__)
    result_path = "../result/size"
    path = os.path.join(abs_path,result_path)
    result_dict = {}
    sizes = [10+10*i for i in range(100)]
    for i in tqdm(range(len(words)),total=len(words)):
        result_dict[words[i]] = {}
        
        txt_file_name = words[i]+'.txt'
        txt_path = os.path.join(path,'txt')
        txt_file_path = os.path.join(txt_path,txt_file_name)

        with open(txt_file_path,'w') as ft:
            ft.write(f"result of the word : {words[i]}\n")
            total = 0
            ok = 0
            for size in sizes:
                total+=1
                make_input = Make_input(text=words[i],font_size=size)
                img, _ = make_input.create_image()
                if words[i] == 'a' and size == 1000:
                    img.show()
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
                decoded_results = processor.tokenizer.batch_decode(outputs.sequences)
                sequence = processor.batch_decode(outputs.sequences)[0]
                sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
                succeed = words[i] == sequence
                if succeed:
                    ok += 1
                ft.write(f"size : {size}px , result (if succeeded?) : {succeed}, output : {sequence} \n")
                result_dict[words[i]][size] = succeed
            ft.write(f"total iteration : {total} , succeed : {ok}\n")

    json_file_name = 'size_change_result.json'
    json_path = os.path.join(path,json_file_name) 

    with open(json_path,'w') as f:
        json.dump(result_dict,f,indent=2)

def get_acc_json():
    abs_path = os.path.dirname(__file__)
    result_path = '../result/size'
    path = os.path.join(abs_path,result_path)
    json_file_name = 'size_change_result.json'
    size_dict = {}
    sizes = [str(10+i*10) for i in range(100)]
    for size in sizes:
        size_dict[size] = 0
    with open(os.path.join(path,json_file_name),'r') as f:
        result_dict = json.load(f)
        for word,dict in result_dict.items():
            for size, ret in dict.items():
                if ret:
                    size_dict[size] += 1
    name = 'size_acc.json'
    with open(os.path.join(path,name),'w') as f:
        json.dump(size_dict,f,indent=2)

def get_cmap_by_acc():
    result_path = '../result/size'
    json_file_name = "size_acc.json"
    result_path = os.path.join(result_path,json_file_name)
    abs_path = os.path.dirname(__file__)
    path = os.path.join(abs_path,result_path)
    acc_dict = {}
    with open(path,'r') as f:
        acc_dict = json.load(f)
    
    acc = []
    for size,cnt in acc_dict.items():
        acc.append(cnt)
    
    size = list(range(10,1010,10))
    save_path = '../result/size'
    fig_name = 'size_acc'
    plt.figure(figsize=(10,10))
    plt.title("accuracy numbers depending on the size of the word")
    plt.xlabel("[px]")
    plt.plot(size,acc)
    plt.savefig(os.path.join(save_path,fig_name))

if __name__ == '__main__':
    #main()
    #get_acc_json()
    get_cmap_by_acc()