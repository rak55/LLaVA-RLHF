import argparse
from transformers import AutoTokenizer, AutoConfig
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import os
from PIL import Image
from io import BytesIO
import re
import requests
import json
from tqdm import tqdm
import torch

#added old funcs.
def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,load_4bit=args.load_4bit)

    dataset = list(read_jsonl(args.dataset))
    demos = list(read_jsonl(args.demos))
    demos = demos[:args.num_demos]
    ex_demos = []
    for d_item in demos:
        ex_demos.append(
            {"image": load_image(os.path.join(args.images_path, d_item["file_name"])),"frame":d_item["frame"],"rationale":d_item["rationale"],"problems":d_item["problems"]})
    answers_file = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    seen_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = json.loads(line)
                seen_ids.add(line["file_name"])
                
    r_prompt = "Reason about whether the posting contains a frame (or more frames), or just states something factual or an experience."
    a_prompt = "If the posting contains a frame, articulate that frame succinctly."
    
    def add_r_turn(conv, question: str, rationale: str | None = None):
        qs = f"{question} {r_prompt}"
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = (
                qs
                + "\n"
                + DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN)
        else:
            qs = qs + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], qs)
        if rationale is not None:
            #rationale = format_rationale(rationale)
            conv.append_message(conv.roles[1], rationale + "\n")
        else:
            conv.append_message(conv.roles[1],None)
        
    def add_a_turn(conv, answer: str | None = None):
        qs = a_prompt
        conv.append_message(conv.roles[0], qs)
        if answer is not None:
            conv.append_message(conv.roles[1],answer + "\n")
        else:
            conv.append_message(conv.roles[1],None)
    
    def run(conv,images):
        prompt = conv.get_prompt()
        input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=[stopping_criteria],
            )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
          print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
          outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
       
    for idx in tqdm(range(len(dataset))):
        if dataset[idx]["file_name"] in seen_ids:
            continue
        ex = dataset[idx]
        image_path = ex["file_name"]
        image = load_image(os.path.join(args.images_path, image_path))
        image_tensor = image_processor.preprocess([d["image"] for d in ex_demos] + [image], return_tensors="pt")["pixel_values"]
        images = image_tensor.half().cuda()
        question = "You will be tasked with identifying and articulating misogyny framings on the social media postings. Each social media posting provided may or may not contain one or more frames of communication."
        "List all the frames and the corresponding reasoning."
        conv = conv_templates[args.conv_mode].copy()
        for d in ex_demos:
            add_r_turn(
                conv,
                question=question,
                rationale=d["rationale"],            #changed near demos. 
            )
            add_a_turn(
                conv,
                answer=d["frame"],                   #change category wrt output.
            )

        final_conv = conv.copy()

        add_r_turn(
            conv,
            question=question,
            
        )
        print(conv.get_prompt())
        rationale = run(conv, images)

        add_r_turn(
            final_conv,
            question=question,
            rationale=rationale,
                                       
        )
        full_conv = final_conv.copy()
        add_a_turn(final_conv)
        pred = run(final_conv, images)
            
        add_a_turn(full_conv, answer=pred)
        
        with open(answers_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": ex["file_name"],
                        "rationale": rationale,
                        "pred": pred,
                    }
                )
                + "\n"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--num_demos", type=int, default=3)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    
    eval_model(args)
