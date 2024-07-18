# check_dataset.py
def check_dataset(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        if not content:
            print("File is empty")
        else:
            try:
                import json
                data = json.loads(content)
                print("File is a valid JSON")
                print(f"Number of entries: {len(data)}")
            except json.JSONDecodeError:
                print("File is not a valid JSON")

if __name__ == "__main__":
    check_dataset('data/pg-wikiSQL-sql-instructions-80k.json')
