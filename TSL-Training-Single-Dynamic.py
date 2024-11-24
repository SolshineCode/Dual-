import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import logging
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptResponseDataset(Dataset):
    """Dataset for prompt-response pairs"""
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize input prompt
        prompt_encoding = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target response
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

class TwoModelTrainer:
    def __init__(
        self,
        static_model_name: str,
        dynamic_model_name: str,
        tokenizer_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize models
        logger.info(f"Loading static model {static_model_name}")
        self.static_model = AutoModelForCausalLM.from_pretrained(static_model_name).to(device)
        self.static_model.eval()  # Set to evaluation mode
        
        logger.info(f"Loading dynamic model {dynamic_model_name}")
        self.dynamic_model = AutoModelForCausalLM.from_pretrained(dynamic_model_name).to(device)
        
        # Freeze the static model
        for param in self.static_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        dynamic_output: torch.Tensor,
        static_output: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the combined loss between dynamic model output -> static model output -> target
        """
        # Loss between static model output and target
        static_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
            static_output.view(-1, static_output.size(-1)),
            target_ids.view(-1)
        )
        
        # Additional loss terms could be added here, such as:
        # - KL divergence between dynamic and static outputs
        # - Auxiliary objectives for the dynamic model
        # - Regularization terms
        
        return static_loss

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_wandb: bool = True
    ):
        """Train the dynamic model"""
        if log_wandb:
            wandb.init(project="two-model-finetuning")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            self.dynamic_model.parameters(),
            lr=learning_rate
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_dataloader)
        )
        
        for epoch in range(num_epochs):
            self.dynamic_model.train()
            total_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Generate output from dynamic model
                dynamic_output = self.dynamic_model(
                    input_ids=batch['prompt_ids'],
                    attention_mask=batch['prompt_mask']
                ).logits
                
                # Pass dynamic output through static model
                with torch.no_grad():
                    static_output = self.static_model(
                        input_ids=torch.argmax(dynamic_output, dim=-1),
                        attention_mask=batch['prompt_mask']
                    ).logits
                
                # Compute loss
                loss = self.compute_loss(
                    dynamic_output,
                    static_output,
                    batch['target_ids'],
                    batch['target_mask']
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
                
                # Gradient accumulation and optimization
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.dynamic_model.parameters(),
                        max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Log metrics
                if log_wandb and step % 100 == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                    })
                
            # Validation loop
            val_loss = self.evaluate(val_dataloader)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Average training loss: {total_loss / len(train_dataloader)}")
            logger.info(f"Validation loss: {val_loss}")
            
            if log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": total_loss / len(train_dataloader),
                    "val_loss": val_loss
                })

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on the validation set"""
        self.dynamic_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                dynamic_output = self.dynamic_model(
                    input_ids=batch['prompt_ids'],
                    attention_mask=batch['prompt_mask']
                ).logits
                
                static_output = self.static_model(
                    input_ids=torch.argmax(dynamic_output, dim=-1),
                    attention_mask=batch['prompt_mask']
                ).logits
                
                loss = self.compute_loss(
                    dynamic_output,
                    static_output,
                    batch['target_ids'],
                    batch['target_mask']
                )
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def save_model(self, path: str):
        """Save the dynamic model"""
        self.dynamic_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

# Example usage
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
    
    trainer = TwoModelTrainer(
        static_model_name="gpt2",  # Replace with your model
        dynamic_model_name="gpt2",  # Replace with your model
        tokenizer_name="gpt2"      # Replace with your tokenizer
    )
    
    # Create datasets
    train_dataset = PromptResponseDataset(train_data, trainer.tokenizer)
    val_dataset = PromptResponseDataset(val_data, trainer.tokenizer)
    
    # Train the model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-5
    )
    
    # Save the trained model
    trainer.save_model("path/to/save/model")

if __name__ == "__main__":
    main()
