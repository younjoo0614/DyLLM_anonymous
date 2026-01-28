import argparse
import json
import torch
import os
from lm_eval.__main__ import cli_evaluate
from dyllm.eval.adapter import DyLLMAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--tasks", type=str, default="gsm8k")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-shot", type=int, default=5)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--ignore-eos", action="store_true", default=False)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--num-full-steps", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--output-file", type=str, default=None)
    ap.add_argument("--log-samples", action="store_true", default=False)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model_args_dict = {
        "model_path": args.model_path,
        "max_new_toks": args.max_new_tokens,
        "tensor_parallel_size": args.tp_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ignore_eos": args.ignore_eos,
        "trust_remote_code": args.trust_remote_code,
        "num_steps": args.num_steps,
        "num_full_steps": args.num_full_steps,
        "block_size": args.block_size,
        "threshold": args.threshold,
    }
    model_args = ",".join([f"{k}={v}" for k, v in model_args_dict.items()])

    cli_args = argparse.Namespace(
        model="dyllm",
        model_args=model_args,
        tasks=args.tasks,
        num_fewshot=args.num_shot,
        batch_size=args.batch_size,
        output_path=args.output_file,
        limit=args.limit,
        device="cuda",
        check_integrity=False,
        write_out=False,
        show_config=False,
        include_path=None,
        gen_kwargs=None,
        verbosity="INFO",
        wandb_args="",
        wandb_config_args="",
        hf_hub_log_args="",
        predict_only=False,
        seed=[0, 1234, 1234, 1234],
        trust_remote_code=args.trust_remote_code,
        confirm_run_unsafe_code=True,
        log_samples=args.log_samples,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        use_cache=None,
        cache_requests=None,
        max_batch_size=None,
        samples=None,
        metadata=None,
    )

    with torch.inference_mode():
        cli_evaluate(cli_args)


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    main()
