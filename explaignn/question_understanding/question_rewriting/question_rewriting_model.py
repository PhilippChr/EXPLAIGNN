import os
import torch
import transformers
from pathlib import Path

import explaignn.question_understanding.question_rewriting.dataset_question_rewriting as dataset


class QuestionRewritingModel(torch.nn.Module):
    def __init__(self, config):
        super(QuestionRewritingModel, self).__init__()
        self.config = config
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(
            "castorini/t5-base-canard"
        )
        self.tokenizer = transformers.T5TokenizerFast.from_pretrained("castorini/t5-base-canard")

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def save(self):
        """Save model."""
        model_path = self.config["qrew_model_path"]
        # create dir if not exists
        model_dir = os.path.dirname(model_path)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self):
        """Load model."""
        state_dict = torch.load(self.config["qrew_model_path"])
        self.model.load_state_dict(state_dict)
        # move to GPU (if possible)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_path, dev_path):
        """Train model."""
        # load datasets
        train_dataset = dataset.DatasetQuestionRewriting(self.config, self.tokenizer, train_path)
        dev_dataset = dataset.DatasetQuestionRewriting(self.config, self.tokenizer, dev_path)
        # arguments for training
        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir="explaignn/question_understanding/question_rewriting/results",  # output directory
            num_train_epochs=self.config[
                "qrew_num_train_epochs"
            ],  # total number of training epochs
            per_device_train_batch_size=self.config[
                "qrew_per_device_train_batch_size"
            ],  # batch size per device during training
            per_device_eval_batch_size=self.config[
                "qrew_per_device_eval_batch_size"
            ],  # batch size for evaluation
            warmup_steps=self.config[
                "qrew_warmup_steps"
            ],  # number of warmup steps for learning rate scheduler
            weight_decay=self.config["qrew_weight_decay"],  # strength of weight decay
            logging_dir="explaignn/question_understanding/question_rewriting/logs",  # directory for storing logs
            logging_steps=1000,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end="True"
            # predict_with_generate=True
        )
        # create the object for training
        trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )
        # training progress
        trainer.train()
        # store model
        self.save()

    def inference(self, inputs):
        """
        Run the model on the given input.
        Snippet taken from: https://github.com/gonced8/rachael-scai/blob/main/demo.py
        """
        # encode
        rewrite_input_ids = self.tokenizer.encode(
            inputs,
            truncation=False,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            rewrite_input_ids = rewrite_input_ids.cuda()
        # generate
        output = self.model.generate(
            rewrite_input_ids,
            max_length=self.config["qrew_max_output_length"],
        )
        # decoding
        model_rewrite = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return model_rewrite
