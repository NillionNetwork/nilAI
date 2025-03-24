/**
 * Test Setup:
 * 1. cd test/e2e
 * 2. npm init -y
 * 3. npm install --save-dev jest axios @jest/globals
 * 4. Add to package.json: { "scripts": { "test": "jest" } }
 * 5. Run in e2e folder: npx jest test_api.test.js
 * 
 * Gitignore:
 * Add to .gitignore:
 * test/e2e/node_modules/
 * test/e2e/package-lock.json
 * test/e2e/coverage/
 * test/e2e/*.log
 */

const axios = require('axios');
const { describe, it, expect } = require('@jest/globals');
const { BASE_URL, AUTH_TOKEN, test_models } = require('./config');

const client = axios.create({
  baseURL: BASE_URL,
  headers: {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${AUTH_TOKEN}`
  },
});

async function streamRequest(endpoint, payload) {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: {
      'accept': 'application/json',
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${AUTH_TOKEN}`
    },
    body: JSON.stringify(payload)
  });
  return response;
}

describe('Test API', () => {
  describe('Basic Endpoints', () => {
    it('should test health endpoint', async () => {
      const response = await client.get('health');
      console.log(response.data);
      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('status');
    });

    it('should test models endpoint', async () => {
      const response = await client.get('/models');
      expect(response.status).toBe(200);
      expect(Array.isArray(response.data)).toBe(true);
      
      const modelNames = response.data.map(model => model.id);
      test_models.forEach(model => {
        expect(modelNames).toContain(model);
      });
    });

    it('should test usage endpoint', async () => {
      const response = await client.get('/usage');
      expect(response.status).toBe(200);
      expect(typeof response.data).toBe('object');
      
      const expectedKeys = ['total_tokens', 'completion_tokens', 'prompt_tokens', 'queries'];
      expectedKeys.forEach(key => {
        expect(response.data).toHaveProperty(key);
      });
    });

    it('should test attestation endpoint', async () => {
      const response = await client.get('/attestation/report');
      expect(response.status).toBe(200);
      expect(typeof response.data).toBe('object');
      expect(response.data).toHaveProperty('cpu_attestation');
      expect(response.data).toHaveProperty('gpu_attestation');
      expect(response.data).toHaveProperty('verifying_key');
    });
  });

  describe('Model Tests', () => {
    test_models.forEach(model => {
      it(`should test standard request for ${model}`, async () => {
        const payload = {
          model,
          messages: [
            { role: 'system', content: 'You are a helpful assistant that provides accurate and concise information.' },
            { role: 'user', content: 'What is the capital of France?' }
          ],
          temperature: 0.2
        };

        const response = await client.post('/chat/completions', payload);
        expect(response.status).toBe(200);
        expect(response.data).toHaveProperty('choices');
        expect(response.data.choices.length).toBeGreaterThan(0);
        
        const content = response.data.choices[0].message.content;
        expect(content).toBeTruthy();
        expect(response.data.choices[0].finish_reason).toBe('stop');
        expect(response.data.usage.prompt_tokens).toBeGreaterThan(0);
        expect(response.data.usage.completion_tokens).toBeGreaterThan(0);
        expect(response.data.usage.total_tokens).toBeGreaterThan(0);
        
        console.log(`\nModel ${model} standard response: ${content.slice(0, 100)}...`);
      });

      it(`should test streaming request for ${model}`, async () => {
        const payload = {
          model,
          messages: [
            { role: 'system', content: 'You are a helpful assistant that provides accurate and concise information.' },
            { role: 'user', content: 'Write a short poem about mountains.' }
          ],
          temperature: 0.2,
          stream: true
        };

        const response = await streamRequest('/chat/completions', payload);
        expect(response.status).toBe(200);
        
        const reader = response.body.getReader();
        let chunkCount = 0;
        let content = '';
        let hadUsage = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const text = new TextDecoder().decode(value);
          const lines = text.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data:')) {
              chunkCount++;
              const chunk = line.slice(5).trim();
              if (chunk) {
                const chunkJson = JSON.parse(chunk);
                if (chunkJson.choices?.[0]?.delta?.content) {
                  content += chunkJson.choices[0].delta.content;
                }
                if (chunkJson.usage) {
                  hadUsage = true;
                }
              }
            }
          }
        }
      
        expect(chunkCount).toBeGreaterThan(0);
        expect(hadUsage).toBe(true);
        console.log(`Received ${chunkCount} chunks for ${model} streaming request`);
      });

      it(`should test tools request for ${model}`, async () => {
        const payload = {
          model,
          messages: [
            { role: 'system', content: 'You are a helpful assistant that provides accurate and concise information.' },
            { role: 'user', content: 'What is the weather like in Paris today?' }
          ],
          temperature: 0.2,
          tools: [{
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Get current temperature for a given location.',
              parameters: {
                type: 'object',
                properties: {
                  location: { type: 'string', description: 'City and country e.g. Paris, France' }
                },
                required: ['location'],
                additionalProperties: false
              },
              strict: true
            }
          }]
        };

        try {
          const response = await client.post('/chat/completions', payload);
          expect(response.status).toBe(200);
          expect(response.data).toHaveProperty('choices');
          expect(response.data.choices.length).toBeGreaterThan(0);

          const message = response.data.choices[0].message;
          if (message.tool_calls) {
            expect(message.tool_calls.length).toBeGreaterThan(0);
            const firstCall = message.tool_calls[0];
            expect(firstCall.function.name).toBe('get_weather');
            const args = JSON.parse(firstCall.function.arguments);
            expect(args.location.toLowerCase()).toContain('paris');
          } else {
            expect(message.content).toBeTruthy();
          }
        } catch (e) {
          console.log(`\nError testing tools with ${model}: ${e.message}`);
          throw e;
        }
      });
    });
  });

  describe('Additional Tests', () => {
    it('should test invalid auth token', async () => {
      const invalidClient = axios.create({
        baseURL: BASE_URL,
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json',
          'Authorization': 'Bearer invalid_token_123'
        }
      });

      try {
        await invalidClient.get('/attestation/report');
      } catch (error) {
        expect([401, 403]).toContain(error.response.status);
      }
    });

    it('should test large payload handling', async () => {
      const largeSystemMessage = 'Hello '.repeat(10000);
      const payload = {
        model: test_models[0],
        messages: [
          { role: 'system', content: largeSystemMessage },
          { role: 'user', content: 'Respond briefly' }
        ],
        max_tokens: 50
      };

      const response = await client.post('/chat/completions', payload);
      expect([200, 413]).toContain(response.status);
      if (response.status === 200) {
        expect(response.data).toHaveProperty('choices');
        expect(response.data.choices.length).toBeGreaterThan(0);
      }
    });
  });
});