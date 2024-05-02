import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from datasets import load_dataset

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.config.output_hidden_states = True
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(model.encoder)

dataset = load_dataset("hf-internal-testing/example-documents", split="test")

image = dataset["image"][0]
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

encoder_last_hidden = outputs.encoder_hidden_states
print(encoder_last_hidden[4].size())

decoded_results = processor.tokenizer.batch_decode(outputs.sequences)
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
output = processor.token2json(sequence)
print("-------------------------------------------")
print("output : ", output)
print("-------------------------------------------")