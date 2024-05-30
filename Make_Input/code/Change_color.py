from Make_Input import Make_input
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

def main():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    #making list of alphabets and numbers
    lower = [chr(ord('a')+i) for i in range(26)] #a~z
    upper = [chr(ord('A')+i) for i in range(26)] #A~Z
    num = [chr(ord('0')+i) for i in range(10)] #0~9
    words = lower+upper+num
    #path
    abs_path = os.path.dirname(__file__)
    result_path = "../result/color"
    path = os.path.join(abs_path,result_path)
    result_dict = {}
    for i in tqdm(range(len(words)),total=len(words)):
        result_dict[words[i]] = {}
        
        txt_file_name = words[i]+'.txt'
        txt_path = os.path.join(path,'txt')
        txt_file_path = os.path.join(txt_path,txt_file_name)
        
        with open(txt_file_path,'w') as ft:
            ft.write(f"result of the word : {words[i]}\n")
            total = 0
            ok = 0
            for a in range(9):
                for b in range(9):
                    for c in range(9):
                        total+=1
                        make_input = Make_input(text=words[i],font_size=90,text_color=(min(32*a,255),min(32*b,255),min(32*c,255)),text_pos=(0,256))
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
                        decoded_results = processor.tokenizer.batch_decode(outputs.sequences)
                        sequence = processor.batch_decode(outputs.sequences)[0]
                        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
                        succeed = words[i] == sequence
                        if succeed:
                            ok += 1
                        ft.write(f"color : ({min(32*a,255)},{min(32*b,255)},{min(32*c,255)}) , result (if succeeded?) : {succeed}, output : {sequence} \n")
                        result_dict[words[i]][str((min(32*a,255),min(32*b,255),min(32*c,255)))] = succeed
            ft.write(f"total iteration : {total} , succeed : {ok}\n")

    json_file_name = 'color_change_result.json'
    json_path = os.path.join(path,json_file_name) 

    with open(json_path,'w') as f:
        json.dump(result_dict,f,indent=2)

def get_acc_json():
    abs_path = os.path.dirname(__file__)
    result_path = '../result/color'
    path = os.path.join(abs_path,result_path)
    json_file_name = 'color_change_result.json'
    color_dict = {}
    colors = []
    for a in range(9):
            for b in range(9):
                for c in range(9):
                    colors.append(str(((min(32*a,255),min(32*b,255),min(32*c,255)))))
    for color in colors:
        color_dict[color] = 0
    
    with open(os.path.join(path,json_file_name),'r') as f:
        result_dict = json.load(f)
        for word,dict in result_dict.items():
            for color, ret in dict.items():
                if ret:
                    color_dict[color] += 1
    name = 'color_acc.json'
    with open(os.path.join(path,name),'w') as f:
        json.dump(color_dict,f,indent=2)

if __name__ == '__main__':
    main()
    get_acc_json()

