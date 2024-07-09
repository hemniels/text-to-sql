from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/sql-query', methods=['POST'])
def sql_query():
    data = request.json
    # Hier würde die Verarbeitung der Textanfrage zu SQL erfolgen
    query = data.get('query')
    return jsonify({'query': query})

if __name__ == '__main__':
    app.run(debug=True)
