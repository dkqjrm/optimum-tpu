from dataclasses import dataclass, field
from typing import Optional
from trl import SFTConfig


@dataclass
class CustomSFTConfig(SFTConfig):
    # SFT-specific parameters with TRL defaults
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Maximum length of the tokenized sequence."}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to group multiple sequences into fixed-length blocks."}
    )
    completion_only_loss: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to compute loss only on the completion part."}
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(
        default=None,
        metadata={"help": "The integration to report results and logs to."},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the run for logging."}
    )
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "The learning rate scheduler to use."}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to adopt during training."},
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "Number of update steps between two evaluations."},
    )
    save_strategy: str = field(
        default="no",
        metadata={"help": "The checkpoint save strategy to adopt during training."},
    )
    save_steps: int = field(
        default=0,
        metadata={"help": "Number of updates steps before two checkpoint saves."},
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Limit the total amount of checkpoints."},
    )
    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation."})
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load the best model found during training at the end."},
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "The metric to use to compare two different models."},
    )
    greater_is_better: bool = field(
        default=False,
        metadata={
            "help": "Whether the metric_for_best_model should be maximized or not."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a specific checkpoint to resume training from."},
    )