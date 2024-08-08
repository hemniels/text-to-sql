from datasets import load_dataset

def load_wikisql_dataset():
    """
    Lädt das WikiSQL-Dataset mithilfe des Hugging Face `datasets`-Moduls.
    """
    dataset = load_dataset("kaxap/pg-wikiSQL-sql-instructions-80k")
    return dataset['train']
