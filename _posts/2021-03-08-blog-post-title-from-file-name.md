## Trying out Bing character recognition capabilities against other OCR Engines

Exploring and trying to understand how GPT handles image input analysis, especially concerning Optical Character Recognition (OCR). Nowadays we almost take this for granted but usually, this task is split into Text Detection and Text Recognition. 

In this blog, we will briefly explore how OCR Engines like JaidedAI's EasyOCR, PaddleOCR, and Tesseract fare against the results obtained from analyzing images in GPT4 through a prompt. 

The images are as follows: 

Image1:


<img width="200" alt="image" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/8c71158b-ef2b-493a-939e-356c95a403bd"><img width="573" alt="bingtext1" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/32ceb3c4-5e37-4a14-83d0-d2ede0c06257">




Image2: 


<img width="200" alt="image" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/a76bdd30-4536-4f17-ba59-d3881d7b2ffd"><img width="569" alt="bingtext2" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/1e866bee-d56a-4d35-a029-6ede7d7f94a3">

From the above descriptions, we see that not only is GPT-4 Vision apt at describing the content of the text but also quite good at describing the state the box is in. 



---

### Testing other OCR Engines

### 1. EasyOCR

EasyOCR follows the two-step process of using CRAFT (Character-Region Awareness For Text detection) for Text detection and Text Recognition to transcribe text. The framework of EasyOCR as shown in its repository is shown below 

<img width="200" alt="image" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/14893598-135c-4c76-abb6-172da1e65df3">

The outputs obtained from the EasyOCR are as follows 

Image1:

<img width="200" alt="image" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/861a6ca5-b495-4f5a-93bc-145041b0648a"><img width="500" alt="EasyOCRUSB" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/2ff471f7-678b-4076-a2af-d960f9fde7d3">


Analysis:
We see bounding boxes outlined in the image output above so we know what text is being transcribed currently, but the confidence relays that the model is not familiar with recognizing superscript annotations, which explains the low confidence score and the semi-correct text.


Image2:

<img width="200" alt="image" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/57857e06-c70c-4198-93ff-54d5883fc648">
<img width="500" alt="Screenshot 2024-01-23 142352" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/26ebb70f-7567-46ea-9489-3c39950002c1">



Analysis:
Again we see bounding boxes outlined in the output image but we start to see discrepancies between the text detection here up against GPT4-Vision, mainly with the singular digits on either side of the barcode. GPT was not able to highlight the numbers, however, EasyOCR was at least able to detect the presence of one of these digits and correctly recognize the text therein. 

### 2.PaddleOCR

Trying an instance of PaddleOCR on hugging face we observe the following results below: 

Image1: 
<img width="180" alt="img1paddle" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/afd1bfe6-283f-4eb2-a292-a5280360bae0">


Image2:
<img width="118" alt="img2paddle" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/3105d017-256e-4e29-a9ad-a6d01ffe30d7">

Analysis: 
We see a similar performance as observed earlier in EasyOCR, where superscript text has been ignored completely, however, text was detected elsewhere where GPT failed to pick up. 

### Reasons GPT4-Vision has a few hiccups with transcribing text 

These reasons can easily be sourced from the blog post about GPT4-Vision where it outlines all the limitations of the model, in this particular case the text might be too small for it to pick up. It cannot properly define the spatial reasoning or establish a proper context for the space between text and thus refuses to show the output to the user. 
We can try and reason that the image was too small, however, the size of the images, in this case, was 1536x2040, so the numbers were distinct enough to catch. 


### Deciphering Handwriting 

To further examine GPT4-Vision's ability to recognize text, I gave it the ultimate task of deciphering my cryptic handwriting, which gave the following output. 

<img width="376" alt="test2" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/6e3db5d7-dc16-443a-8e81-d3c263deae7a">

Which it deciphers excellently: 

<img width="300" alt="handwritinggpt4" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/502e3f2a-e2c0-484b-b4ec-d0f60e3d0ab2">

Putting this up against results obtained from OCR Engines trained for Handwriting, we observe the following:

### 1.Adobe Scan
Adobe actually has a solution where you can scan any document, convert it to a PDF, which translates its contents to transcribed text when highlighted and copied. 

You'll have to take my word for it but the output I got was "Aame s Vanspelli " which is a far cry from what I wrote.

### 2.TrOCR
TrOCR is Microsoft's end-to-end Transformer-based OCR model for text recognition with pre-trained CV and NLP models, this is quoted from a brief summary of the paper listed at paperswithcode.com

The following code was executed to obtain a result 

```tsql
!pip install -q transformers
```

```
import requests
from PIL import Image

url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open("/content/test2.png").convert("RGB")
image

from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# calling the processor is equivalent to calling the feature extractor
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

<img width="200" alt="handwritingoutput" src="https://github.com/vijayvanapalli96/vijay.github.io/assets/46009628/0a9b3ab6-8f2c-499e-b42b-a1dd44ac7e13">


We see a more coherent result, since the the OCR engine may be mapping the handwriting to whole words rather than going letter by letter, the last name is incorrect and it fails to distinguish two words separately.

From our observations, we can conclude that when it comes to Hand Written Character Recognition GPT4-Vision is superior and more flexible. 

We do need to look at the resources expended to obtain these results. 

According to [https://lifestyle.livemint.com/news/big-story/ai-carbon-footprint-openai-chatgpt-water-google-microsoft-111697802189371.html#:~:text=The%20estimated%20energy%20consumption%20of%20a%20ChatGPT%2D4%20query%20is,and%20number%20of%20tokens%20processed.](https://shorturl.at/vwJP4)

The estimated energy consumption of a ChatGPT-4 query is 0.001-0.01 kWh (3.6-36 kJ), depending on the model size and number of tokens processed. Whereas the size of this OCR model would be much smaller and more efficient to run on lower-end systems thus making it more accessible for applications. 

