from flask import Flask, request, jsonify
from fullTextSearch import fullTextSearch
import os

app = Flask(__name__)
csv_filepath = os.path.join(os.path.dirname(__file__), 'chuyen_khoan.csv')
fts = fullTextSearch(filename=csv_filepath)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Transaction Full Text Search API"}), 200

@app.route('/search', methods=['POST'])
def search_transactions():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query"}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({"error": "Do not let query empty"}), 400
    
    ranked_docs = fts.search(query)

    results = []
    for doc_id, score in ranked_docs:
        transaction_info = fts.get_transaction_info(doc_id)
        if transaction_info:
            results.append({
                **transaction_info 
            })
    
    return jsonify({
        "query": query,
        "results": results
    }), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)