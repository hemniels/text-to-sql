<script>
    import { onMount } from 'svelte';
    import { sendQuery } from './api.js';

    let query = '';
    let result = '';

    const handleQuery = async () => {
        result = await sendQuery(query);
    }
</script>

<main>
    <h1>Text-to-SQL App</h1>
    <textarea bind:value={query} placeholder="Gib deine Anfrage ein..."></textarea>
    <button on:click={handleQuery}>Absenden</button>
    <pre>{result}</pre>
</main>

<style>
    main {
        text-align: center;
        margin: 0 auto;
        max-width: 600px;
    }
    textarea {
        width: 100%;
        height: 100px;
    }
    button {
        margin: 10px 0;
    }
</style>
