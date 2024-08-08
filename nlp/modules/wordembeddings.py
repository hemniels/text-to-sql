# Laden des WikiSQL-Datasets
from datasets import load_dataset
wikisql_dataset = load_dataset("wikisql")

# Trainieren des Word2Vec-Modells auf dem WikiSQL-Dataset
from gensim.models import Word2Vec

# Extrahiere SQL-Statements
sql_statements = [query['human_readable'] for query in wikisql_dataset['train']['sql']]

# Preprocessing: Text in Token-Listen umwandeln
processed_statements = [simple_preprocess(statement) for statement in sql_statements]

# Trainieren des Word2Vec-Modells
w2v_model = Word2Vec(sentences=processed_statements, vector_size=50, window=5, min_count=1, workers=4)
