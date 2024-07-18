import dataset
from scripts.loader import loader as ld
from torch.utils.data import DataLoader

class DataModules:
    """
    Test Class to run tests on Data Operations
    1. DataLoading Script - nlp/loader
    2. -
    """

if __name__ == "__main__":
    dataset_url = "kaxap/pg-wikiSQL-sql-instructions-80k"
    ds = ld.dataset(dataset_url)
    # glove_path = loader.download_glove(glove_path, 50)
