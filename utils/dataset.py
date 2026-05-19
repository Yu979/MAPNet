import torch
from torch.utils.data import Dataset

class HybridDataset(Dataset):
    def __init__(self, data_list, mol_tokenizer, prot_tokenizer, config):
        self.data = data_list
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.max_mol_len = config.get('data.max_mol_len', 50)
        self.max_prot_len = config.get('data.max_prot_len', 30)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles, sequence, label = self.data[idx]
        
        mol_inputs = self.mol_tokenizer(
            smiles, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_mol_len, 
            return_tensors="pt"
        )
        
        prot_inputs = self.prot_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_prot_len, 
            return_tensors="pt"
        )
        
        return {
            'mol_ids': mol_inputs['input_ids'].squeeze(0),
            'mol_mask': mol_inputs['attention_mask'].squeeze(0),
            'prot_ids': prot_inputs['input_ids'].squeeze(0),
            'prot_mask': prot_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor([label], dtype=torch.float)
        }