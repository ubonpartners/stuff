import os
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
import re
import hashlib
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor

def generate_token_padding(target_tokens=600, tokens_per_line=14):
    pad_line = "This sentence is included only to increase the token count."
    num_lines = target_tokens // tokens_per_line
    return "\n".join([f"{i+1}. {pad_line}" for i in range(num_lines)])

class simple_llm:
    def __init__(self, model="gpt-4o-mini", cost_opt=True):
        self.openai=None
        self.genai=None
        self.cost_opt=cost_opt
        if model.startswith("gpt"):
            import openai as openai
            self.openai=openai
            self.client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
            self.image_size=512
        elif model.startswith("gemini"):
            from google import genai
            from google.genai import types
            self.genai=genai
            self.types=types
            self.client=self.genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.image_size=384

        self.prompt_tokens=0
        self.completion_tokens=0
        self.cached_prompt_tokens=0
        self.inferences=0
        self.refusals=0
        self.exceptions=0
        self.system_prompt=None
        self.system_prompt_cache=None

        if model=="gpt-4o-mini":
            # https://platform.openai.com/docs/pricing
            self.model=model
            self.cost_prompt=0.15
            self.cost_cached_prompt=0.075
            self.cost_output=0.60
        elif model=="gpt-4o":
            self.model=model
            self.cost_prompt=2.50
            self.cost_cached_prompt=1.25
            self.cost_output=10.00
        elif model=="gpt-4.1-nano":
            self.model=model
            self.cost_prompt=0.10
            self.cost_cached_prompt=0.025
            self.cost_output=0.40
        elif model=="gpt-4.1-mini":
            self.model=model
            self.cost_prompt=0.40
            self.cost_cached_prompt=0.10
            self.cost_output=1.60
        elif model=="gpt-4.1":
            self.model=model
            self.cost_prompt=2.00
            self.cost_cached_prompt=0.50
            self.cost_output=8.00
        elif model=="gpt-5":
            self.model=model
            self.cost_prompt=1.25
            self.cost_cached_prompt=0.125
            self.cost_output=10.00
        elif model=="gpt-5-mini":
            self.model=model
            self.cost_prompt=0.25
            self.cost_cached_prompt=0.025
            self.cost_output=2.00
        elif model=="gpt-5-nano":
            self.model=model
            self.cost_prompt=0.05
            self.cost_cached_prompt=0.005
            self.cost_output=0.40
        elif model=="gemini-2.5-pro":
            self.model=model
            self.cost_prompt=1.25
            self.cost_cached_prompt=0.31
            self.cost_output=10.0
        elif model=="gpt-5.4":
            self.model=model
            self.cost_prompt=2.5
            self.cost_cached_prompt=0.25
            self.cost_output=15.0
        elif model=="gpt-5.4-mini":
            self.model=model
            self.cost_prompt=0.75
            self.cost_cached_prompt=0.075
            self.cost_output=4.50
        elif model=="gemini-2.5-flash":
            # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash
            # https://cloud.google.com/vertex-ai/generative-ai/pricing#gemini-models-2.5
            # https://ai.google.dev/gemini-api/docs/pricing
            self.model=model
            self.cost_prompt=0.30
            self.cost_cached_prompt=0.075
            self.cost_output=2.5
        elif model=="gemini-3.1-flash-lite-preview":
            self.model=model
            self.cost_prompt=0.25
            self.cost_cached_prompt=0.025
            self.cost_output=1.5
        elif model=="gemini-3-flash-preview":
            self.model=model
            self.cost_prompt=0.5
            self.cost_cached_prompt=0.05
            self.cost_output=3
        elif model=="gemini-2.5-flash-lite":
            self.model="gemini-2.5-flash-lite-preview-06-17"
            self.cost_prompt=0.10
            self.cost_cached_prompt=0.025
            self.cost_output=0.4
        else:
            assert False, f"Unknown LLM model {model}"

        if self.cost_opt and "gpt-5" in model:
            self.cost_prompt*=0.5
            self.cost_cached_prompt*=0.5
            self.cost_output*=0.5

        #print(f"Initialized LLM {model}")

    def set_system_prompt(self, system_prompt):
        self.system_prompt=system_prompt
        if self.genai:
            time_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
            while True:
                total_tokens = self.client.models.count_tokens(model=self.model, contents=system_prompt)
                if total_tokens.total_tokens>2048:
                    break
                system_prompt=generate_token_padding(target_tokens=2048-total_tokens.total_tokens)+system_prompt
                total_tokens = self.client.models.count_tokens(model=self.model, contents=system_prompt)
            print(f"Caching {total_tokens}")
            system_prompt=generate_token_padding()+system_prompt
            self.system_prompt_cache = self.client.caches.create(
                model=self.model,
                config=self.types.CreateCachedContentConfig(
                    display_name=f"ubonsystemprompt-{time_suffix}", # used to identify the cache
                    #system_instruction=(
                    #    system_prompt
                    #),
                    contents=[system_prompt],
                    ttl="300s",
                )
            )

    def infer(self, prompt, images=None, attempts=1):
        refusal=None
        if self.openai is not None:
            if self.cost_opt:
                if attempts<7:
                    attempts=7
            images_b64=[]
            if images is not None:
                for i in images:
                    if isinstance(i, str):
                        images_b64.append(i)
                    else:
                        images_b64.append(base64.b64encode(i).decode("utf-8"))

            content=[]
            content.append({"type": "text", "text": prompt})
            for i in images_b64:
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{i}", "detail": "low"},
                                })
            messages=[]
            if self.system_prompt is not None:
                messages.append({"role": "system","content": [{"type": "text","text": self.system_prompt}]})
            messages.append({"role": "user", "content": content})
            for attempt in range(attempts):
                try:
                    params = dict(
                        model=self.model,
                        messages=messages,
                        seed=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        max_completion_tokens=2048,
                        user="UbonPartnersUsers",
                    )
                    params["timeout"]=180.0 # 3 minutes
                    if self.model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                        params["reasoning_effort"]="minimal"
                    if not "gpt-5" in self.model:
                        params["temperature"]=0.0
                    if self.cost_opt and "gpt-5" in self.model:
                        params["service_tier"]="flex"

                    completion = self.client.chat.completions.create(**params)

                    usage=completion.usage
                    ret=completion.choices[0].message.content
                    refusal=completion.choices[0].message.refusal
                    if refusal is None:
                        self.inferences+=1
                        self.prompt_tokens+=usage.prompt_tokens
                        self.completion_tokens+=usage.completion_tokens
                        self.cached_prompt_tokens+=usage.prompt_tokens_details.cached_tokens
                        #print(f"Tokens {usage.prompt_tokens} {usage.prompt_tokens_details.cached_tokens}" )
                        break # sucess
                    else:
                        self.refusals+=1
                        print(f"OpenAI refusal: {refusal}")
                except (self.openai.APITimeoutError, self.openai.APIConnectionError, self.openai.RateLimitError, self.openai.InternalServerError) as e:
                    print(f"OpenAI {e.__class__.__name__}, retrying... (attempt {attempt+1}/{attempts})")
                    if attempt < attempts - 1:
                        time.sleep(2 ** attempt)  # exponential backoff
                    else:
                        raise
                except Exception as e:
                    self.exceptions+=1
                    print(f"OpenAI exception: {e}")
                ret=""

        elif self.genai is not None:
            from google.genai.types import GenerateContentConfig
            #if system_prompt is not None:
            #    prompt=system_prompt+prompt
            contents=[prompt]
            if images is not None:
                for i in images:
                    contents.append(self.types.Part.from_bytes(data=i, mime_type='image/jpeg'))

            for attempt in range(attempts):
                try:
                    if self.system_prompt_cache is not None:
                        response = self.client.models.generate_content(
                            model=self.model,
                            contents=contents,
                            config=GenerateContentConfig(
                                cached_content=self.system_prompt_cache.name,
                                temperature=0.2,
                                seed=123
                            )
                    )
                    else:
                        response = self.client.models.generate_content(
                            model=self.model,
                            contents=contents,
                            config=GenerateContentConfig(
                                system_instruction=[self.system_prompt],
                                temperature=0.2,
                                seed=123
                            )
                        )
                    ret=response.text
                    metadata=response.usage_metadata
                    self.inferences+=1
                    self.prompt_tokens+=metadata.prompt_token_count
                    if metadata.cached_content_token_count is not None:
                        self.cached_prompt_tokens+=metadata.cached_content_token_count
                    self.completion_tokens+=metadata.candidates_token_count
                    break
                except Exception as e:
                    print(f"Gemini_request error: {e}")
                    ret=""
        return ret

    def get_base64_jpeg(self, path, quality=85, optimize=False):
        with Image.open(path) as im:
            # Ensure JPEG-compatible mode
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            # Downscale in-place, preserving aspect ratio
            im.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            # Encode to JPEG in memory
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=optimize)
            jpeg_bytes = buf.getvalue()
        return base64.b64encode(jpeg_bytes).decode("ascii")

    def get_stats(self):
        uncached_cost=self.cost_prompt*(self.prompt_tokens-self.cached_prompt_tokens)
        cached_cost=self.cost_cached_prompt*self.cached_prompt_tokens
        output_cost=self.cost_output*self.completion_tokens
        cost=(uncached_cost+cached_cost+output_cost)

        return {"model":self.model,
                "inferences":self.inferences,
                "prompt_tokens":self.prompt_tokens,
                "completion_tokens":self.completion_tokens,
                "cached_prompt_tokens":self.cached_prompt_tokens,
                "cost($)":f"{cost/1000000:8.6f}",
                "cost/Minf($)":f"{cost/(self.inferences+0.001):6.2f}",
                "cost%-prompt-uncached":(100.0*uncached_cost)/(cost+1e-7),
                "cost%-prompt-cached":(100.0*cached_cost)/(cost+1e-7),
                "cost%-output":(100.0*output_cost)/(cost+1e-7),
                #"cost-output":output_cost,
                #"cost-output-rate":self.cost_output,
                "cost-optimised":self.cost_opt,
                "refusals":self.refusals,
                "exceptions":self.exceptions}

    def _cache_dir_for(self, cache_name: str | None) -> Path:
        """
        Normalize cache_name and return the cache directory path.
        Only allow [A-Za-z0-9._-] and replace everything else (including "/") with "_".
        """
        if not cache_name:
            cache_name = "default"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", cache_name)
        return Path("/mldata/llm_cache") / safe

    @staticmethod
    def _hash_inputs(j, system_prompt: str) -> str:
        """
        Create a compact, order-sensitive hash from prompt, images list (or None), and system_prompt.
        Uses blake2b with an 8-byte digest for short filenames; bump if you want fewer collisions.
        """
        h = hashlib.blake2b(digest_size=8)
        sep = b"\0"
        # prompt
        h.update(b"P"); h.update(j["prompt"].encode("utf-8")); h.update(sep)
        # system prompt
        h.update(b"S"); h.update(system_prompt.encode("utf-8")); h.update(sep)
        # images (list of base64 strings or None)
        if "img_uid" in j and j["img_uid"] is not None:
            h.update(b"I"); h.update(j["img_uid"].encode("utf-8")); h.update(sep)
        h.update(b"I0"); h.update(sep)
        return h.hexdigest()

    def prepare_job(self, j):
        assert "cache_name" in j
        if not "cache_dir" in j:
            cache_dir = self._cache_dir_for(j["cache_name"])
            cache_dir.mkdir(parents=True, exist_ok=True)
            j["cache_dir"]=cache_dir
        if not "b64_image" in j:
            if "img_path" in j:
                j["b64_image"]=self.get_base64_jpeg(j["img_path"])
        img_uid=j.get("img_uid", None)
        if img_uid is None:
            if "b64_image" in j:
                j["img_uid"]=j["b64_image"]
        if not "digest" in j:
            j["digest"]=self._hash_inputs(j, getattr(self, "system_prompt", ""))
        if not "cache_file" in j:
            j["cache_file"] = j["cache_dir"] / f"{j['digest']}.txt"

    def infer_cached(self, j, attempts=1, should_hit=False):
        self.prepare_job(j)
        cache_file=j["cache_file"]
        try:
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8")
        except Exception:
            # If read fails, fall through to recompute
            pass

        if should_hit:
            print("WARING: llm infer cache- should have hit!")

        # Miss: compute and write atomically
        result = self.infer(j["prompt"], images=[j["b64_image"]], attempts=attempts)

        try:
            # Write atomically to avoid partial files if multiple workers
            with NamedTemporaryFile("w", delete=False, dir=j["cache_dir"], encoding="utf-8") as tmp:
                tmp.write(result if isinstance(result, str) else str(result))
                tmp_path = Path(tmp.name)
            tmp_path.replace(cache_file)  # atomic on POSIX
        except Exception:
            # If caching fails, still return the result
            return result

        return result

    def process_parallel(self, j):
        r=self.infer_cached(j, attempts=2)
        return r

    def infer_cached_batch(self, jobs, attempts=1, num_parallel=512):
        # a "job" should be a dictionary
        # prompt: prompt to use
        # cache_name: folder /mldata/llm_cache/<cache_name> will be used to cache the LLM request results
        # optional- image_path OR b64_img : image to use with prompt
        # optional- image_uid: key to use for img in cache if not present, will use hash(b64 of img jpeg)

        to_run=[]
        for j in jobs:
            self.prepare_job(j)
            if not os.path.isfile(j["cache_file"]):
                to_run.append(j)

        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            _ = list(executor.map(self.process_parallel, to_run))

        # now we can run again, it should be 100% certain all results are cached and
        # there shold be no work to do
        ret=[]
        for j in jobs:
            ret.append(self.infer_cached(j, attempts=attempts, should_hit=True))
        return ret
