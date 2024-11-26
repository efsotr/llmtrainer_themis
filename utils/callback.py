from transformers import TrainerCallback, TrainerState, TrainerControl, IntervalStrategy

from .arguments import MaxTrainingArguments

class EndEvalCallback(TrainerCallback):
    def on_step_end(self, args: MaxTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (
            args.extra_save_steps is not None 
            and args.extra_save_steps > 0
            and state.global_step % args.extra_save_steps == 0
        ):
            control.should_save = True

        # Log & Evaluate & Save
        if state.global_step >= state.max_steps:
            if args.logging_strategy != IntervalStrategy.NO:
                control.should_log = True
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True
            if args.evaluation_strategy != IntervalStrategy.NO:
                control.should_evaluate = True

        if args.check_stage == "ck_run":
            if state.global_step >= 3:
                control.should_training_stop = True
                control.should_save = False
                if args.evaluation_strategy != IntervalStrategy.NO:
                    control.should_evaluate = True

        return control