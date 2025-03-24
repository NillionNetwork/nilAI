// test/e2e/config.js
const ENVIRONMENT = process.env.ENVIRONMENT || 'dev';
const AUTH_TOKEN = process.env.AUTH_TOKEN;

if (!AUTH_TOKEN) {
  throw new Error('AUTH_TOKEN environment variable is required');
}

let BASE_URL;
if (ENVIRONMENT === 'dev') {
  BASE_URL = 'http://localhost:8080/v1';
} else if (ENVIRONMENT === 'ci') {
  BASE_URL = 'http://127.0.0.1:8080/v1';
} else if (ENVIRONMENT === 'mainnet') {
  BASE_URL = 'https://nilai-e176.nillion.network/v1';
} else {
  throw new Error(`Invalid environment: ${ENVIRONMENT}`);
}

const models = {
  mainnet: [
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
  ],
  testnet: [
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
  ],
  ci: [
    'meta-llama/Llama-3.2-1B-Instruct',
  ],
};

const test_models = models[ENVIRONMENT];

module.exports = {
  ENVIRONMENT,
  AUTH_TOKEN,
  BASE_URL,
  test_models,
};