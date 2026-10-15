# PBD orchestration (archived copies from Vision-OPD-setup/, cluster-specific paths)
- orch_pbd.sbatch.tmpl / orch_off6kb.sbatch.tmpl: hold-based orchestrators (smoke -> train -> eval -> record+trim); orch_o6PBDk.sbatch = the official-6K Keep run.
- eval_off6k_par.sh (multi-GPU, splits ckpts over GPUs) -> eval_off6k_w.sh (per worker: merge, vLLM serve, OPD-Aha eval/infer.py, gpt-oss-120b judge rule-first + pure-llm); eval_off6k.sh = single-GPU original.
- trim_off6k.sh + record_run.py: once V*/ZoomBench/HR-4K/HR-8K are complete, write run_records/<tag>.yaml and keep only the ZoomBench-best ckpt.
- watchdog_pbd.sbatch: disk guard (drop optimizer states), restart dead orchestrators, re-request vanished holds, GPU-idle log, release holds when runs finish.
