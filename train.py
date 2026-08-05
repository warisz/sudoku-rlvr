import random

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from sudoku import generate_all_valid, generate_puzzle, extract, verify

MODEL = "Qwen/Qwen3-1.7B"
# BLANKS = [2, 3, 4]  
N_PER_BLANK = 400         # training puzzles per difficulty
SEED = 1234           


def build_prompt(puzzle):
    """Must match the prompt used in eval, or before/after isn't comparable."""
    return f"""
        Solve this 4x4 sudoku. Fill zeroes with digits 1-4 so each row, column, and 2x2 box contains 1,2,3,4 exactly once. Here is a 2D array, in which zeroes represent empty spaces, and every item in the first dimension is a complete row:

        {puzzle}

        Respond with ONLY the solved 2D array and nothing else. Do not add an explanation. Example format: [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]]
        """

def make_dataset():
    random.seed(SEED)
    boards = generate_all_valid()
    rows = []
    for n_blanks in range(1,16): 
        for _ in range(N_PER_BLANK):
            board = random.choice(boards)
            puzzle = generate_puzzle(board, n_blanks)
            rows.append({
                "prompt": [{"role": "user", "content": build_prompt(puzzle)}],
                "puzzle": puzzle,          # extra column, passed to reward fn
            })
    random.shuffle(rows) #dont want model to get used to one difficulty
    return Dataset.from_list(rows)


#trl lib will grab one row from make_dataset output and then run the desired number of rollouts (completions), then pass into this function
def reward_fn(completions, puzzle, **kwargs):
    """TRL calls this with a batch of rollouts (completions). Return one float per completion.

    `completions` is a list of rollout outputs; `puzzle` is the matching list from
    the dataset column of the same name, all one puzzle .
    """
    print("=" * 40)
    print("RAW COMPLETION:", repr(completions[0]))
    text = completions[0][0]["content"]
    print("TEXT:", repr(text[:200]))
    print("EXTRACTED:", extract(text))
    print("PUZZLE:", puzzle[0])
    print("VERIFY:", verify(puzzle[0], extract(text)))
    print("=" * 40)
    
    rewards = []
    for completion, p in zip(completions, puzzle):
        text = completion[0]["content"]
        rewards.append(1.0 if verify(p, extract(text)) else 0.0)

    print("REWARDS:", rewards)
    return rewards


def main():
    dataset = make_dataset()
    print(f"training on {len(dataset)} puzzles")

    config = GRPOConfig(
        output_dir="grpo-sudoku",
        num_generations=4,          # rollouts per prompt (i.e. 4 ROLLOUTS PER PUZZLE)
        temperature=1.0,             # must be >0 or rollouts are identical
        max_completion_length=128, # token limit per rollout
        per_device_train_batch_size=8,   # multiple of num_generations. this is how many rollout outputs sit on the GPU at a time, per batch. If num_generations is 4, and this param is 8, then we can hold the rollouts for 2 puzzles worth on the GPU 
        gradient_accumulation_steps=4, #number of batches before a weight update. So if it's set to 4, then we will generate 4 batches worth of rollouts and then update the weight

        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        report_to="none",            # set "wandb" if you want curves logged
        chat_template_kwargs={"enable_thinking": False}
        # max_steps=3
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=MODEL,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model("grpo-sudoku/final")


if __name__ == "__main__":
    main()