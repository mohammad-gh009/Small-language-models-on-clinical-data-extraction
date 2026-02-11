import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import time

set_seed(42)
torch.random.manual_seed(42)

split_base = "aya_expanse"
loca_path = "CHANGE THIS PATH"
batch_size =  # use costum batch based on the GPU

model_name = "CohereLabs/aya-expanse-8b"
# "meta-llama/Llama-3.2-3B-Instruct",
# "google/gemma-3-1b-it",
# "Qwen/Qwen2.5-1.5B-Instruct",
# "Qwen/Qwen2.5-7B-Instruct",
# "meta-llama/Llama-3.1-8B-Instruct",
# "openai/gpt-oss-20b"




pipe = pipeline("text-generation", model=model_name, device="cuda", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset = load_from_disk("path/to/saved_dataset")
dset_all = dataset["train"]

pipe.tokenizer.pad_token_id = (
    pipe.model.config.eos_token_id
)  # [0] # uncomment this for Llama and gemma models. 
pipe.tokenizer.padding_side = "left"


def generate_batch(texts):
    outputs = pipe(
        texts,
        max_new_tokens=512,
        do_sample=False,
        batch_size=batch_size,
        return_full_text=False,
    )
    torch.cuda.empty_cache()
    return [out[0]["generated_text"] for out in outputs]


def generate_all(dset, split_name):
    all_results = []
    prompts = dset["prompt"]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}"
        )

        batch_results = generate_batch(batch_prompts)
        all_results.extend(batch_results)

    final_set = dset.add_column("outputs", all_results)

    print(f"Generated {len(all_results)} results")
    final_set.save_to_disk(loca_path, split=split_name)


start = time.time()
generate_all(dset_all, split_base)
end = time.time()
print(end - start)
