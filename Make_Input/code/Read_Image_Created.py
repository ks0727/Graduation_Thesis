from Make_Input import Make_input
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import os

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.no_grad():
    for i in range(2):
        for j in range(500,1138,1):
            model.encoder.encoder.layers[3].blocks[i].intermediate.dense.weight[j] = 0.0

model.to(device)
dataset = load_dataset("naver-clova-ix/cord-v2", split="test")

make_input = Make_input(text='あいうえおかきくけこさしすせそ',font_size=90,text_pos=(0,256))
img, _ = make_input.create_image()
abs_path = os.path.dirname(__file__)

#result_path = os.path.join(result_path,file_name)
#img.save(os.path.join(abs_path,result_path))

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

output = processor.token2json(sequence)
print("-------------------------------------------")
print("output : ", output)
print("-------------------------------------------")