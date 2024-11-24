import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Literal
import logging
import wandb
from dataclasses import dataclass
from enum import Enum, auto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingMode(Enum):
    TRAIN_DYNAMIC = auto()
    TRAIN_STATIC = auto()
    TRAIN_BOTH = auto()

@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    training_mode: TrainingMode = TrainingMode.TRAIN_BOTH
    switch_frequency: int = 1  # How often to switch which model is training (in batches)
    log_wandb: bool = True

class PromptResponseDataset(Dataset):
    """Dataset for prompt-response pairs"""
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt_encoding = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            item['target'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'prompt_ids': prompt_encoding['input_ids'].squeeze(),
            'prompt_mask': prompt_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_mask': target_encoding['attention_mask'].squeeze()
        }

class AlternatingModelTrainer:
    def __init__(
        self,
        model_a_name: str,
        model_b_name: str,
        tokenizer_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize both models
        logger.info(f"Loading model A: {model_a_name}")
        self.model_a = AutoModelForCausalLM.from_pretrained(model_a_name).to(device)
        
        logger.info(f"Loading model B: {model_b_name}")
        self.model_b = AutoModelForCausalLM.from_pretrained(model_b_name).to(device)
        
        # Initialize optimizers for both models
        self.optimizer_a = optim.AdamW(self.model_a.parameters(), lr=5e-5)
        self.optimizer_b = optim.AdamW(self.model_b.parameters(), lr=5e-5)
        
        self.current_training_model = TrainingMode.TRAIN_DYNAMIC

    def switch_training_model(self):
        """Switch which model is being trained"""
        if self.current_training_model == TrainingMode.TRAIN_DYNAMIC:
            self.current_training_model = TrainingMode.TRAIN_STATIC
            logger.info("Switching to train static model")
        else:
            self.current_training_model = TrainingMode.TRAIN_DYNAMIC
            logger.info("Switching to train dynamic model")

    def compute_loss(
        self,
        output_a: torch.Tensor,
        output_b: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute losses for both models
        """
        # Loss for model A
        loss_a = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
            output_a.view(-1, output_a.size(-1)),
            target_ids.view(-1)
        )
        
        # Loss for model B
        loss_b = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
            output_b.view(-1, output_b.size(-1)),
            target_ids.view(-1)
        )
        
        return loss_a, loss_b

    def forward_pass(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass through both models
        """
        # Get the currently active and frozen models based on training mode
        if self.current_training_model == TrainingMode.TRAIN_DYNAMIC:
            active_model = self.model_b
            frozen_model = self.model_a
        else:
            active_model = self.model_a
            frozen_model = self.model_b

        # Forward pass through active model
        with torch.set_grad_enabled(training):
            output_active = active_model(
                input_ids=batch['prompt_ids'],
                attention_mask=batch['prompt_mask']
            ).logits

        # Forward pass through frozen model
        with torch.no_grad():
            output_frozen = frozen_model(
                input_ids=torch.argmax(output_active, dim=-1),
                attention_mask=batch['prompt_mask']
            ).logits

        return output_active, output_frozen

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """
        Perform a single training step
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get the active optimizer
        optimizer = self.optimizer_b if self.current_training_model == TrainingMode.TRAIN_DYNAMIC else self.optimizer_a
        
        # Forward pass
        output_active, output_frozen = self.forward_pass(batch, training=True)
        
        # Compute losses
        loss_a, loss_b = self.compute_loss(
            output_active,
            output_frozen,
            batch['target_ids'],
            batch['target_mask']
        )
        
        # Use the appropriate loss based on which model is being trained
        loss = loss_b if self.current_training_model == TrainingMode.TRAIN_DYNAMIC else loss_a
        
        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        # Optimizer step with gradient clipping
        if config.gradient_accumulation_steps == 1:
            torch.nn.utils.clip_grad_norm_(
                self.model_b.parameters() if self.current_training_model == TrainingMode.TRAIN_DYNAMIC 
                else self.model_a.parameters(),
                config.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()
        
        return {
            "loss": loss.item(),
            "loss_a": loss_a.item(),
            "loss_b": loss_b.item()
        }

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainingConfig
    ):
        """Train both models with alternating updates"""
        if config.log_wandb:
            wandb.init(project="alternating-model-training")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size
        )
        
        # Training loop
        for epoch in range(config.num_epochs):
            total_loss_a = 0
            total_loss_b = 0
            steps = 0
            
            for step, batch in enumerate(train_dataloader):
                # Switch training model if needed
                if step % config.switch_frequency == 0:
                    self.switch_training_model()
                
                # Training step
                losses = self.train_step(batch, config)
                total_loss_a += losses["loss_a"]
                total_loss_b += losses["loss_b"]
                steps += 1
                
                # Logging
                if config.log_wandb and step % 100 == 0:
                    wandb.log({
                        "step": step,
                        "loss_a": losses["loss_a"],
                        "loss_b": losses["loss_b"],
                        "training_model": self.current_training_model.name
                    })
            
            # Validation
            val_losses = self.evaluate(val_dataloader)
            
            # Logging
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            logger.info(f"Average training loss A: {total_loss_a / steps}")
            logger.info(f"Average training loss B: {total_loss_b / steps}")
            logger.info(f"Validation loss A: {val_losses['loss_a']}")
            logger.info(f"Validation loss B: {val_losses['loss_b']}")
            
            if config.log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss_a": total_loss_a / steps,
                    "train_loss_b": total_loss_b / steps,
                    "val_loss_a": val_losses["loss_a"],
                    "val_loss_b": val_losses["loss_b"]
                })

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate both models"""
        self.model_a.eval()
        self.model_b.eval()
        total_loss_a = 0
        total_loss_b = 0
        steps = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output_a, output_b = self.forward_pass(batch, training=False)
                loss_a, loss_b = self.compute_loss(
                    output_a,
                    output_b,
                    batch['target_ids'],
                    batch['target_mask']
                )
                total_loss_a += loss_a.item()
                total_loss_b += loss_b.item()
                steps += 1
        
        return {
            "loss_a": total_loss_a / steps,
            "loss_b": total_loss_b / steps
        }

    def save_models(self, path_a: str, path_b: str):
        """Save both models"""
        self.model_a.save_pretrained(path_a)
        self.model_b.save_pretrained(path_b)
        self.tokenizer.save_pretrained(path_a)
        self.tokenizer.save_pretrained(path_b)

def main():
    # Sample training data
    train_data = [
        {"prompt": "Input text 1", "target": "Target text 1"},
        {"prompt": "Input text 2", "target": "Target text 2"},
    ]
    val_data = [
        {"prompt": "Val input 1", "target": "Val target 1"},
        {"prompt": "Val input 2", "target": "Val target 2"},
    ]
    
    # Initialize trainer
    trainer = AlternatingModelTrainer(
        model_a_name="gpt2",
        model_b_name="gpt2",
        tokenizer_name="gpt2"
    )
    
    # Create datasets
    train_dataset = PromptResponseDataset(train_data, trainer.tokenizer)
    val_dataset = PromptResponseDataset(val_data, trainer.tokenizer)
    
    # Configure training
    config = TrainingConfig(
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-5,
        switch_frequency=100  # Switch models every 100 batches
    )
    
    # Train the models
    trainer.train(train_dataset, val_dataset, config)
    
    # Save the trained models
    trainer.save_models("path/to/save/model_a", "path/to/save/model_b")

if __name__ == "__main__":
    main()
