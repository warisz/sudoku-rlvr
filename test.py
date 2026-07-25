from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).to("mps")

puzzle_prompt = """Solve this 4x4 sudoku. Fill blanks (.) with digits 1-4 so each row,
column, and 2x2 box contains 1,2,3,4 exactly once. Here is a 2D array, in which 0's represent empty spaces, 
and every item in the first dimension is a complete row. 

Here's the incomplete sudoku you need to solve:
[[3,0,0,0],[0,2,0,0],[0,0,4,1],[0,0,0,4]]

ONLY Output the solved 2D array which has replaced all 0's with valid numbers ranging from 1-4. DO NOT output anything other than the 2d array. """

msgs = [{"role": "user", "content": puzzle_prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").to("mps")

out = model.generate(**ids, max_new_tokens=512, do_sample=True, temperature=1.0) 
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))