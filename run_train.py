from optimum.tpu import fsdp_v2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import sys
from peft import LoraConfig
from trl import SFTTrainer
from config import CustomSFTConfig
import torch_xla.core.xla_model as xm
from callbacks import LoRAOnlyCallback
from custom_dataset import load_single_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_train.py <config.json>")
        sys.exit(1)
    
    config = load_config(sys.argv[1])
    
    fsdp_v2.use_fsdp_v2()
    
    model_id = config["llm_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Fix Qwen3 chat template for assistant_only_loss support
    if tokenizer.chat_template and "{% generation %}" not in tokenizer.chat_template:
        print("🔧 Fixing Qwen3 chat template for assistant_only_loss support...")
        
        original_template = tokenizer.chat_template
        
        # Find and replace the assistant-specific section
        # Look for: {%- elif message.role == "assistant" %}
        # Add generation right after <|im_start|>assistant\n is output
        
        # Replace the entire assistant block with generation-wrapped version
        assistant_block_old = """{%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\"name\": \"' }}
                {{- tool_call.name }}
                {{- '\", \"arguments\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}"""
        
        assistant_block_new = """{%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {% generation %}
        {%- if reasoning_content %}
            {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' }}
        {%- endif %}
        {{- content.lstrip('\\n') }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\"name\": \"' }}
                {{- tool_call.name }}
                {{- '\", \"arguments\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {% endgeneration %}
        {{- '<|im_end|>\\n' }}"""
        
        fixed_template = original_template.replace(assistant_block_old, assistant_block_new)
        
        # Handle add_generation_prompt case
        fixed_template = fixed_template.replace(
            "{{- '<|im_start|>assistant\\n' }}",
            "{{- '<|im_start|>assistant\\n' }}{% generation %}"
        )
        
        tokenizer.chat_template = fixed_template
        print("✅ Fixed Qwen3 chat template with {% generation %}/{% endgeneration %} blocks")
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    
    # Load train and eval datasets
    train_data = load_single_dataset(config["train_dataset"], tokenizer)
    eval_data = load_single_dataset(config["eval_dataset"], tokenizer) if config.get("eval_dataset") else None

    fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
    fsdp_training_args["fsdp_config"]["min_num_params"] = 0
    
    lora_config = LoraConfig(
        lora_alpha=config.get("lora_alpha", 128),
        lora_dropout=config.get("lora_dropout", 0.05),
        r=config.get("lora_r", 256),
        bias=config.get("lora_bias", "none"),
        target_modules=config.get("lora_target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )
    
    # Create training args from config
    training_args = CustomSFTConfig(
        per_device_train_batch_size=config.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        max_steps=config.get("max_steps", -1),
        num_train_epochs=config.get("num_train_epochs", 3),
        output_dir=config["output_dir"],
        optim="adafactor",
        logging_steps=config.get("logging_steps", 10),
        dataloader_drop_last=config.get("dataloader_drop_last", True),
        save_strategy=config.get("save_strategy", "no"),
        save_steps=config.get("save_steps", None),
        save_total_limit=config.get("save_total_limit", None),
        assistant_only_loss=config.get("assistant_only_loss", False),
        report_to=config.get("report_to"),
        run_name=config.get("run_name"),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 100),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        max_length=config.get("max_length", 1024),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        dataloader_prefetch_factor=config.get("dataloader_prefetch_factor", 4),
        dataloader_pin_memory=config.get("dataloader_pin_memory", True),
        dataloader_persistent_workers=config.get("dataloader_persistent_workers", True),
        remove_unused_columns=config.get("remove_unused_columns", False),
        seed=config.get("seed", 42),
        resume_from_checkpoint=config.get("resume_from_checkpoint"),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 1000),
        do_eval=config.get("do_eval", False),
        load_best_model_at_end=config.get("load_best_model_at_end", False),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=config.get("greater_is_better", False),
        **fsdp_training_args,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args,
        peft_config=lora_config,
    )
    
    # Add LoRA callback
    lora_save_steps = config.get("lora_save_steps")
    lora_save_epochs = config.get("lora_save_epochs")
    if lora_save_steps or lora_save_epochs:
        trainer.add_callback(LoRAOnlyCallback(steps=lora_save_steps, epochs=lora_save_epochs))
    
    trainer.train()
    
    # Save final adapter
    final_output_dir = os.path.join(config["output_dir"], "final-adapter")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False, save_adapter=True)
    trainer.tokenizer.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()