import torch
import litgpt
from litgpt import LLM
from litgpt.data import DeceptivePatterns
import lightning as L
import dotenv

dotenv.load_dotenv()


class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None):
        super().__init__()

        self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)
        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)

    def training_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]


def main(model_name="Qwen/Qwen2.5-3B-Instruct"):
    import os

    lit_model = LitLLM(checkpoint_dir=model_name)
    data = DeceptivePatterns(access_token=os.getenv("HF_TOKEN"))

    data.connect(lit_model.llm.tokenizer, batch_size=8, max_seq_length=10*1024)

    trainer = L.Trainer(
        devices=2,
        accelerator="cuda",
        max_epochs=2,
        accumulate_grad_batches=4,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.generate("hello world")
    lit_model.llm.save("finetuned_checkpoint")


if __name__ == "__main__":
    main()