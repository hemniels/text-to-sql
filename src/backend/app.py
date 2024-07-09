from flask import Flask, request, jsonify
from nlp import text_to_sql

app = Flask(__name__)

@app.route('/api/sql-query', methods=['POST'])
def sql_query():
    data = request.json
    text_query = data.get('query')
    sql_query = text_to_sql(text_query)
    return jsonify({'query': sql_query})

if __name__ == '__main__':
    app.run(debug=True)
