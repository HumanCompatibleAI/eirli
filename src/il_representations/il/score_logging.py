"""SB3 score-logging callback for MAGICAL (but should be safe to add to include
when using any environment---if the desired `eval_score` key is not in `infos`,
then it won't add any log entries)."""

from stable_baselines3.common.callbacks import BaseCallback


class SB3ScoreLoggingCallback(BaseCallback):
    """Callback for SB3 RL algorithms which extracts the 'eval_score' from the
    step info dict (if it exists) and includes it in the logs. Useful for
    MAGICAL, which reports end-of-trajectory performance using `eval_score`.

    Tested for PPO, but may work for other algorithms too."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._scores = None
        self._has_score_key = False

    def _on_rollout_start(self):
        self._scores = []

    def _on_step(self):
        assert self._scores is not None, \
            "_on_step() called before _on_rollout_start()"

        infos = self.locals.get('infos')
        dones = self.locals.get('dones')
        if infos is None or len(infos) == 0 or dones is None \
           or len(dones) == 0 or len(infos) != len(dones):
            # PPO should pass this check; other algs may use different names
            raise ValueError(
                "expected to find non-empty, equal-length `infos` and `dones` "
                "in local scope of SB3 algorithm, but infos={infos} and "
                "dones={dones}.  Are you actually using PPO?")

        eval_scores = [info.get('eval_score') for info in infos]
        if not all(s is not None for s in eval_scores):
            if self._has_score_key or any(s is not None for s in eval_scores):
                raise ValueError(
                    "environment provides eval_score for some steps but not "
                    "all")
            return

        self._has_score_key = True
        self._scores.extend(
            score for score, done in zip(eval_scores, dones) if done
        )

    def _on_rollout_end(self):
        if self._has_score_key:
            for s in self._scores:
                logger = self.locals.get('self').logger
                logger.record_mean("eval_score", s)
        else:
            # this indicates a bug: we should not be adding scores if we can't
            # find the key!
            assert not self._scores, \
                "no score key detected, but self._scores={self._scores!r}"
