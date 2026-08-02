from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).to("cuda")

# REQUIRED for batched decoder-only generation
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def run_model(input_puzzles):
    """Take a LIST of puzzles, return a LIST of output strings (same order)."""
    prompts = []
    for input_puzzle in input_puzzles:
        puzzle_prompt = f"""
        Solve this 4x4 sudoku. Fill zeroes with digits 1-4 so each row,
        column, and 2x2 box contains 1,2,3,4 exactly once. Here is a 2D array, in which zeroes represent empty spaces, 
        and every item in the first dimension is a complete row:

        {input_puzzle}

        Respond with ONLY the solved 2D array and nothing else. Do not add an explanation. Example format: [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]]
        """
        msgs = [{"role": "user", "content": puzzle_prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
        prompts.append(text)

    ids = tok(prompts, return_tensors="pt", padding=True).to("cuda") #tokenizes all the prompts at once 
    out = model.generate(**ids, max_new_tokens=512, do_sample=False)  #returned as tokens with original prompts

    prompt_len = ids["input_ids"].shape[1]
    return tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]