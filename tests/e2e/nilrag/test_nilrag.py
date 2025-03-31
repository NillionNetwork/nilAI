import json
import time

from nilrag.config import load_nil_db_config
from nilrag.nildb_requests import ChatCompletionConfig

from nilrag.util import (
    create_chunks,
    encrypt_float_list,
    generate_embeddings_huggingface,
    load_file,
)

import nilql
from ..config import BASE_URL, AUTH_TOKEN, test_models


DEFAULT_CONFIG = "examples/nildb_config.json"
DEFAULT_FILE_PATH = "examples/data/20-fake.txt"


async def initialize_schema_and_query(config_file):
    """
    Initialize schema and query for nilDB nodes.

    This script:
    1. Loads the nilDB configuration from a JSON file
    2. Generates JWT tokens for authentication
    3. Creates a schema for storing embeddings and chunks
    4. Creates a query for computing differences between embeddings
    5. Updates the configuration file with the generated IDs and tokens
    """
    # Load NilDB configuration
    nil_db, secret_key = load_nil_db_config(config_file, require_secret_key=True)
    jwts = nil_db.generate_jwt(secret_key, ttl=3600)  # type: ignore
    print(nil_db)
    print()

    # Upload encrypted data to nilDB
    print("Initializing schema...")
    start_time = time.time()
    schema_id = await nil_db.init_schema()
    end_time = time.time()
    print(f"Schema initialized successfully in {end_time - start_time:.2f} seconds")

    print("Initializing query...")
    start_time = time.time()
    diff_query_id = await nil_db.init_diff_query()
    end_time = time.time()
    print(f"Query initialized successfully in {end_time - start_time:.2f} seconds")

    # Update config file with new IDs and tokens
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for node_data, jwt in zip(data["nodes"], jwts):
        node_data["schema_id"] = schema_id
        node_data["diff_query_id"] = diff_query_id
        node_data["bearer_token"] = jwt
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("Updated nilDB configuration file with schema and query IDs.")


async def upload_data(config_file, file_path):
    """
    Upload data to nilDB using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Initializes encryption keys for different modes
    3. Processes the input file into chunks and embeddings
    4. Encrypts the data using nilQL
    5. Uploads the encrypted data to nilDB nodes
    """

    # Load NilDB configuration
    nil_db, _ = load_nil_db_config(
        config_file,
        require_bearer_token=True,
        require_schema_id=True,
    )
    print(nil_db)
    print()

    # Initialize secret keys for different modes of operation
    num_nodes = len(nil_db.nodes)
    additive_key = nilql.ClusterKey.generate({"nodes": [{}] * num_nodes}, {"sum": True})
    xor_key = nilql.ClusterKey.generate({"nodes": [{}] * num_nodes}, {"store": True})

    # Load and process input file
    paragraphs = load_file(file_path)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)

    # Generate embeddings and chunks
    print("Generating embeddings and chunks...")
    start_time = time.time()
    embeddings = generate_embeddings_huggingface(chunks)
    end_time = time.time()
    print(f"Embeddings and chunks generated in {end_time - start_time:.2f} seconds!")

    # Encrypt chunks and embeddings
    print("Encrypting data...")
    start_time = time.time()
    chunks_shares = [nilql.encrypt(xor_key, chunk) for chunk in chunks]
    embeddings_shares = [
        encrypt_float_list(additive_key, embedding) for embedding in embeddings
    ]
    end_time = time.time()
    print(f"Data encrypted in {end_time - start_time:.2f} seconds")

    # Upload encrypted data to nilDB
    print("Uploading data...")
    start_time = time.time()
    await nil_db.upload_data(embeddings_shares, chunks_shares)  # type: ignore
    end_time = time.time()
    print(f"Data uploaded in {end_time - start_time:.2f} seconds")


async def query_nilai(config_file, prompt):
    """
    Query nilDB with NilAI using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Creates a chat completion configuration
    3. Sends the query to nilAI with nilRAG
    4. Displays the response and timing information
    """

    # Load NilDB configuration
    nil_db, _ = load_nil_db_config(
        config_file,
        require_bearer_token=True,
        require_schema_id=True,
        require_diff_query_id=True,
    )
    print(nil_db)
    print()

    print("Query nilAI with nilRAG...")
    start_time = time.time()
    config = ChatCompletionConfig(
        nilai_url=BASE_URL,
        token=AUTH_TOKEN,
        messages=[{"role": "user", "content": prompt}],
        model=test_models[0],
        temperature=0.2,
        max_tokens=2048,
        stream=False,
    )
    response = nil_db.nilai_chat_completion(config)

    assert response is not None
    assert response["choices"][0]["message"]["content"] is not None
    assert response["choices"][0]["message"]["content"] != ""
    assert "Florence" in response["choices"][0]["message"]["content"]
    end_time = time.time()
    print(json.dumps(response, indent=4))
    print(f"Query took {end_time - start_time:.2f} seconds")


async def test_nilrag():
    await initialize_schema_and_query("tests/e2e/nildb/config.json")
    await upload_data("tests/e2e/nildb/config.json", "cities.txt")
    await query_nilai(
        "tests/e2e/nildb/config.json",
        "Which city is famous for the Renaissance and capital of Tuscany?",
    )
