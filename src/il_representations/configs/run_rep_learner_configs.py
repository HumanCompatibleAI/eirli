from il_representations import algos
from il_representations.algos.utils import LinearWarmupCosine


def make_run_rep_learner_configs(represent_ex):
    @represent_ex.named_config
    def cosine_warmup_scheduler():
        algo_params = {
            "scheduler": LinearWarmupCosine,
            "scheduler_kwargs": {'warmup_epoch': 2, 'T_max': 10}
        }
        _ = locals()
        del _

    @represent_ex.named_config
    def ceb_breakout():
        env_id = 'BreakoutNoFrameskip-v4'
        train_from_expert = True
        algo = algos.FixedVarianceCEB
        batches_per_epoch = 5
        n_epochs = 1
        _ = locals()
        del _

    @represent_ex.named_config
    def expert_demos():
        dataset_configs = [{'type': 'demos'}]
        _ = locals()
        del _

    @represent_ex.named_config
    def random_demos():
        dataset_configs = [{'type': 'random'}]
        _ = locals()
        del _