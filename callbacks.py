from transformers import TrainerCallback
import torch
import os
import torch_xla.core.xla_model as xm


class LoRAOnlyCallback(TrainerCallback):
    def __init__(self, steps=None, epochs=None):
        self.steps = steps
        self.epochs = epochs
        
    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.steps and state.global_step and state.global_step % self.steps == 0:
            self._save_lora_checkpoint(args, state, model, tokenizer)
        return control
    
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.epochs and state.epoch and int(state.epoch) % self.epochs == 0:
            self._save_lora_checkpoint(args, state, model, tokenizer)
        return control
    
    def _save_lora_checkpoint(self, args, state, model, tokenizer):
        xm.mark_step()
        path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}-adapter")
        if xm.is_master_ordinal(local=False):
            os.makedirs(path, exist_ok=True)
        xm.rendezvous("saving_checkpoint")
        if xm.is_master_ordinal(local=False):
            cpu_state = {}
            for n, p in model.named_parameters():
                if "lora" in n:
                    clean = (n.replace("_orig_module.base_model.model.", "base_model.model.")
                             .replace("._orig_module.", ".").replace(".default.", "."))
                    cpu_state[clean] = p.cpu().detach()
            torch.save(cpu_state, os.path.join(path, "adapter_model.bin"))
            model.peft_config[model.active_adapter].save_pretrained(path)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
        xm.rendezvous("checkpoint_saved")