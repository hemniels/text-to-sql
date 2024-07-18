from dataset import SQLDataset
from torch.utils.data import DataLoader

# Test for SQLDataset class 

def get_dataloader(data_path, glove_path, batch_size=32):
    dataset = SQLDataset(data_path, glove_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    data_loader = get_dataloader('data/pg-wikiSQL-sql-instructions-80k.json', 'data/glove.6B/glove.6B.50d.txt')
    for batch in data_loader:
        print(batch)
        break