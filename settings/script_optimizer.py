from components.optimizer import PromptOptimizer
# from utils.llm_client import SPO_LLM
from utils.llm_client_hf import SPO_LLM

if __name__ == "__main__":
  # Initialize LLM settings
  SPO_LLM.initialize(
    optimize_kwargs={"model": "Qwen/Qwen2.5-0.5B-Instruct", "temperature": 0.7},
    evaluate_kwargs={"model": "Qwen/Qwen2.5-0.5B-Instruct", "temperature": 0.3},
    execute_kwargs={"model": "Qwen/Qwen2.5-0.5B-Instruct", "temperature": 0},
    mode = "base_model"
  )

  # Create and run optimizer
  optimizer = PromptOptimizer(
    optimized_path="workspace",  # Output directory
    initial_round=1,  # Starting round
    max_rounds=2,  # Maximum optimization rounds
    template="nli4ct_evidence.yaml",  # Template file
    name="nli4ct_evidence",  # Project name
  )

  optimizer.optimize()