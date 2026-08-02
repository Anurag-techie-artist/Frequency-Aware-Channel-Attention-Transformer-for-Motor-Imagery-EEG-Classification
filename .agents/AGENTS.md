# Project Rules & Workflow Instructions

## Execution Constraints
- **Do not automatically execute `scripts/train.py` or training scripts after code changes.**
- **Validation step after patches**: Run only unit tests (`pytest` / `unittest`) and static validation.
- **Runtime checks**: If a runtime check is strictly required, run at most **1 or 2 batches** or a **single forward pass**.
- **Manual User Execution**: All profiling, smoke tests, and full training runs will be executed manually by the user in their WSL CUDA environment.
