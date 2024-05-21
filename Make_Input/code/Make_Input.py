from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from typing import Tuple
import random
"""
text : texts to draw in the image
font_size : font size[px] used for texts
font : which kinds of font to use
image_size : size of the output image
text_color : (r,g,b) value of the text
bg_color : (r,g,b) value of the back ground of the image
test_pos: left top position of the text
"""
class Make_input:
    def __init__(self,text:str='A',
                font_size:int=50,
                font:str='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                image_size:Tuple[int,int]=(2560,1920),
                text_color:Tuple[int,int,int]=(0,0,0),
                bg_color:Tuple[int,int,int]=(255,255,255),
                text_pos:Tuple[int,int]=(1000,200)) -> None:
        self.text = text
        self.font_size = font_size
        self.font = font
        self.image_size = image_size
        self.text_color = text_color
        self.bg_color = bg_color
        self.text_pos = text_pos

    #funciton to make the image that includes the text specified
    def create_image(self)->Tuple[Image.Image,str]:
        img = Image.new('RGB',self.image_size,self.bg_color)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.font,self.font_size)
        font2 = ImageFont.truetype(self.font,30)
        text_width,test_height = draw.textsize(self.text,font=font)
        draw.text(self.text_pos,self.text,fill=self.text_color,font=font)

        return img,self.text
    
    #function to make the image whose texts are completely generated by random
    def generate_random_text(self,include_sign:bool=False):
        img = Image.new('RGB',self.image_size,self.bg_color)
        draw = ImageDraw.Draw(img)
        max_word_len = 50
        max_sequence = 18
        sequence = []
        seq_len = random.randint(1,max_sequence)
        if include_sign:
            en = [chr(i) for i in range(ord('!'),ord('~')+1)]
        else:
            en_lower = [chr(ord('a')+i) for i in range(26)]
            en_upper = [chr(ord('A')+i) for i in range(26)]
            numbers = [chr(ord('0')+i) for i in range(10)]
            en = en_lower + en_upper+numbers

        non_use_hiragana = ['ゐ','ゑ','ゔ']
        non_use_katakana = ['ヰ','ヱ','ヴ','ヷ','ヸ','ヹ','ヺ']
        non_use_hiragana = [ord(x) for x in non_use_hiragana]
        non_use_katakana = [ord(x) for x in non_use_katakana]
        hiragana = [chr(i) for i in range(ord('あ'),ord('ゖ')) if i not in non_use_hiragana]
        katakana = [chr(i) for i in range(ord('ア'),ord('ヶ')) if i not in non_use_katakana]

        words_option = en+hiragana+katakana
        for _ in range(seq_len):
            word = ""
            word_len = random.randint(1,max_word_len)
            for _ in range(word_len):
                p = random.random()
                if p <= 0.2:
                    word += ' '
                else: 
                    word += random.choice(words_option)
            #print(word)
            sequence.append(word)
        
        now_h = 0
        for i in range(len(sequence)):
            font_sz = random.randint(30,60)
            font = ImageFont.truetype(self.font,font_sz)
            draw.text((random.randint(0,400),now_h),sequence[i],fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255)),font=font)
            now_h += font_sz
            now_h += random.randint(10,100)
            if(now_h > 1920):
                break
        print(img,sequence)
        return img,sequence
    
    

