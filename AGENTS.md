# cartridges

This is an implementation of the Cartridges paper from HazyResearch.

## Setup

The long drawn reference implementation is in @cartridges_ref (from https://github.com/HazyResearch/cartridges). 

1. Use uv for managing the python environment and packages.
2. Check if a uv environment called `.venv` is present. If not, create it by running `uv venv .venv`. Use uv to install the dependencies.
3. Understand the logic from the paper first. Cartridges (Eyuboglu et al., 2025) takes an end-to-end approach: it gradient-optimizes a compact cache directly against downstream loss by backpropagating through the frozen LLM, essentially doing at inference time, for every new context, what we do once at training time.
4. Use the alphaxiv mcp server to fetch the paper contents and use the deepwiki mcp server to answer any questions from the reference repo - https://github.com/HazyResearch/cartridges
4. Use it and the reference implementation to implement a standalone clean implementation. Use the data section from the code to understand how this implementation can be tested and proven to show compression and advantages over naive implementation.
5. Keep the logic and code clean and elegant.
6. Use a small model like Qwen/Qwen3-4B for the training implementation.


## Experimentation

Use the cartridges_ref only as a reference. We need to implement the components for training, data, inference cleanly in the current folder ourselves. The implementation should be much simpler and cleaner than the ref implementation. Make sure all the important components are present and working.

Run the experiments end to end to verify the implementation and the compression and memory savings.

The experiment should run on a single GPU. So you need to check that you have a free GPU or a GPU with enough memory available and no *active* processes on it (i.e. no GPU memory utilization). You can not use GPUs 6 and 7 though.

Use the `.venv` venv to run the experiments (in the parent directory). Install whatever is needed to run the code.

Use wandb in online mode for the experiments. For smoke tests and debugging use the offline mode.

Use vllm for inference server of the parent model.

## Output format

The inference script should prove the benefits of the compression using cartridges. 

LOOP FOREVER:

1. You stop when we can show the advantages of using Cartriges as mentioned in the paper. 

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run into issues, look up papers, issues online, etc. The loop runs until the human interrupts you, period.