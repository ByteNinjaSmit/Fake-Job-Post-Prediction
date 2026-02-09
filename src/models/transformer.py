import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BERTClassificationModel:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def tokenize(self, texts, max_length=256):
        return self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
    
    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load_model(self, path):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

if __name__ == "__main__":
    bert = BERTClassificationModel()
    print(bert.model)
