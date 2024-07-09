export async function sendQuery(query) {
    const response = await fetch('/api/sql-query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
    });

    const data = await response.json();
    return data.query;
}
