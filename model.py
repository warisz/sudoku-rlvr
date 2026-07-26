from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).to("mps")

def run_model(input_puzzle):
    puzzle_prompt = f"""Solve this 4x4 sudoku. Fill zeroes with digits 1-4 so each row,
    column, and 2x2 box contains 1,2,3,4 exactly once. Here is a 2D array, in which zeroes represent empty spaces, 
    and every item in the first dimension is a complete row:

    {input_puzzle}

    ONLY Output the solved 2D array which has replaced all zeroes with valid numbers ranging from 1-4. Ensure that the array is outputted. """


    msgs = [{"role": "user", "content": puzzle_prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("mps")

    out = model.generate(**ids, max_new_tokens=512, do_sample=True) 
    print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
