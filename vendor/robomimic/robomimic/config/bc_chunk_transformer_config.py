"""Config for BC Chunk Transformer algorithm."""

from robomimic.config.bc_config import BCConfig


class BCChunkTransformerConfig(BCConfig):
    ALGO_NAME = "bc_chunk_transformer"

    def train_config(self):
        super(BCChunkTransformerConfig, self).train_config()
        # Need full action chunks for supervision.
        self.train.seq_length = 10
        self.train.frame_stack = 10

    def algo_config(self):
        super(BCChunkTransformerConfig, self).algo_config()

        # Force transformer policy in chunk mode.
        self.algo.transformer.enabled = True
        self.algo.transformer.context_length = 10
        self.algo.transformer.supervise_all_steps = True
        self.algo.transformer.pred_future_acs = False

        # Chunk-policy specific knobs.
        self.algo.chunk.enabled = True
        self.algo.chunk.pred_len = 10
        self.algo.chunk.exec_len = 10
        self.algo.chunk.supervise_mode = "current_to_future"
        self.algo.chunk.loss_reduction = "mean_over_time"
