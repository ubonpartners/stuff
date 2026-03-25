# some old code I should port into llm.py one day...
"""
import base64
import time
from concurrent.futures import ThreadPoolExecutor

def anthropic_request(a):
    client=a["client"]
    b64_image=a["b64_image"]
    prompt=a["prompt"]
    attempts=0
    while attempts<3:
        attempts+=1
        try:
            message = client.messages.create(
                model="claude-3-7-sonnet-latest", #"claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system="You are seasoned analyst that studies images and provides accurate structured information.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content" : "The completed json response is:"
                    }
                ]
            )
            break
        except Exception as e:
            error_dict = e.response.json()
            print(f"Anthropic_request error: {e}")
            if error_dict['error']['type']=="rate_limit_error":
                print(f"RETRY {attempts}")
                time.sleep(attempts*15)
                continue

            print(error_dict)
            return ""

    return message.content[0].text

class LLMAnthropic:
    def __init__(self):
        try:
            import anthropic
            anthropic_ok=True
        except:
            anthropic_ok=False
        assert anthropic_ok, "Try pip install anthropic, and set ANTHROPIC_API_KEY"
        self.client = anthropic.Anthropic()
        self.num_parallel=16

    def get_batch(self):
        return 32

    def get_max_size(self):
        return 512,512

    def get_stats(self):
        return {}

    def generate_attributes(self, attrs, jpegs):

        b64_images=[]
        for j in jpegs:
            b64_images.append(base64.b64encode(j).decode('utf-8'))

        prompt="Study the central person in the image and return JSON reponse with the "
        prompt+="following keys describing image attribue. Each key should have boolean value."
        prompt+="If an attribute is present return true. "
        prompt+="If an attribute is NOT present, or if it cannot be determined, please return false."
        prompt+="For the color attributes, they are asking if those color feature prominently in the"
        prompt+=" top or bottom halves of the persons clothing, respectively.\n"
        prompt+="Key list:\n"

        for a in attrs:
            prompt+=a+", "

        s=[]
        for b in b64_images:
            a={"client":self.client, "prompt":prompt, "b64_image":b}
            s.append(a)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            responses = list(executor.map(anthropic_request, s))

        return responses

import base64
from concurrent.futures import ThreadPoolExecutor

def together_request(a):
    client=a["client"]
    b64_image=a["b64_image"]
    attrs=a["attrs"]

    system_prompt="You are an AI that analyses images and provides accurate structured information."

    prompt=""
    prompt+="Study the central person in the image very carefully and return JSON reponse with the "
    prompt+="keys chosen from the list below."
    prompt+="Be careful to be very accurate."
    #prompt+="Study the persons head if visible to decide if they have a hat and their hair length."
    prompt+="For the color attributes pick the closest match in the person's clothing color."
    prompt+="Check your responses are accurate."
    #prompt+="Try hard to analyse the image even if it somewhat unclear or blurry. If you really can't "
    #prompt+="please reply with just 'too blurry'."
    #prompt+="Be accurate; if unsure about an attribute, or if the relevant part of the person "
    #prompt+="is not visible, the answer should be false."


    prompt+="\nKey list:\n"
    for a in attrs:
        if ":" in a:
            a=a.split(":")[1]
        prompt+=a+", "

    m2={
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt
            }
        ]
    }

    m={
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}",
                }
            }
        ]
    }

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        #model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", # Input: $0.27 | Output: $0.85
        #model="meta-llama/Llama-4-Scout-17B-16E-Instruct", # Input: $0.18 | Output: $0.59
        #model="Qwen/Qwen2.5-VL-72B-Instruct",
        messages=[m2, m],
        seed=0,
        temperature=0.2,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=1000
    )

    r=response.choices[0].message.content
    print(f"Response {r}")

    return {"response":r}

class LLMTogether:
    def __init__(self):
        try:
            from together import Together
            together_ok=True
        except ImportError:
            together_ok=False
        assert together_ok, "Try pip install together"
        self.client = Together()
        self.num_parallel=16
        self.inferences=0

    def get_batch(self):
        return 2048

    def get_max_size(self):
        # From https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/#-suggested-system-prompt-
        # We apply a dynamic image transformation strategy that divides the input image into 336×336 pixel tiles.
        # Additionally, a global tile (created by resizing the entire input image to 336×336 pixels) is appended
        # after the local tiles to provide a global view of the input image.
        return 336,336

    def get_stats(self):
        return {}

    def generate_attributes(self, attrs, jpegs):

        b64_images=[]
        for j in jpegs:
            b64_images.append(base64.b64encode(j).decode('utf-8'))
        s=[]
        for b in b64_images:
            a={"client":self.client, "b64_image":b, "attrs":attrs}
            s.append(a)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            responses = list(executor.map(together_request, s))
        del s

        ret=[response["response"] for response in responses]
        for _ in responses:
            self.inferences+=1
        return ret

# Overall              TP=126.0  FP=20.4 FN=17.5  p=0.860 r=0.878 F=0.869 # scout
# Overall              TP=125.0  FP=24.4 FN=18.5  p=0.837 r=0.871 F=0.853 # maverick
# Overall              TP=132.0  FP=31.4 FN=11.5  p=0.808 r=0.920 F=0.860 # maverick
# Overall              TP=111.0  FP=30.4 FN=32.5  p=0.785 r=0.774 F=0.779 # Qwen/Qwen2.5-VL-72B-Instruct
"""