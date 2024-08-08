# Gesamtes System zur Verarbeitung eines Benutzer-Prompts und Generierung einer SQL-Abfrage
def main_workflow(user_prompt, database):
    # 0. Prompt Handling
    tokenized_prompt = handle_prompt(user_prompt)

    # 1. Word Embeddings
    embedded_prompt = [w2v_model.wv[token] for token in tokenized_prompt]

    # 2. Seq2SQL - Vorhersage der SQL-Komponenten
    aggregate = aggregate_predictor(embedded_prompt)
    select_column = select_predictor(embedded_prompt)
    where_clause = where_predictor(embedded_prompt)

    # 2.1 - 2.3: Zusammensetzen der SQL-Abfrage
    sql_query = f"SELECT {aggregate}({select_column}) WHERE {where_clause}"

    # 3. RL Net Ergebnis prüfen
    reward = reinforcement_learning_model(sql_query, database)

    # 3.1: Handling korrekter/inkorrekter Antworten
    if reward > threshold:
        print("SQL query is correct:", sql_query)
    else:
        print("SQL query is incorrect. Adjusting the model...")

    return sql_query
