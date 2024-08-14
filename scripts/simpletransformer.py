from transformers import T5Tokenizer
from text2sql.utils import create_transformer_dataloaders, load_pg_wikiSQL_dataset
from text2sql.models import SimpleSqlTransformerNet, SimpleSqlClassifier


def main():
    # Laden des Datasets
    dataset = load_pg_wikiSQL_dataset()
    
    # Initialisierung des Tokenizers
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Erstellen der Dataloaders
    dataloader = create_transformer_dataloaders(dataset, tokenizer=tokenizer)

    # Initialisierung des SQL Transformer Netzwerks
    sql_transformer_net = SimpleSqlTransformerNet(model_name='t5-small')
    
    # Training des Modells
    sql_transformer_net.train(dataloader, output_dir='./models/t5', epochs=1)
    
    # Initialisierung des SQL Classifiers
    sql_classifier = SimpleSqlClassifier()
    
    # Beispiel-Frage
    question = "Show all orders greater than 100 dollars"
    
    # Klassifikation mit BERT
    aggregate_pred = sql_classifier.classify(question, 'aggregate')
    select_pred = sql_classifier.classify(question, 'select')
    where_pred = sql_classifier.classify(question, 'where')
    
    print("Aggregate Prediction:", aggregate_pred)
    print("Select Prediction:", select_pred)
    print("Where Prediction:", where_pred)
    
    # Generierung der SQL-Abfrage mit T5
    generated_sql = sql_transformer_net.generate_sql(question)
    print("Generated SQL Query:", generated_sql)

if __name__ == "__main__":
    main()