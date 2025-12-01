-- Seed test data for development
-- This migration inserts test users, admin keys, and test credentials

-- Insert admin key
INSERT INTO admins (key, user_id, is_active) VALUES
    ('n i l l i o n', 'admin', true)
ON CONFLICT (key) DO NOTHING;

-- Insert test users
INSERT INTO users (user_id, balance) VALUES
    ('Docs User', 10000.0)
ON CONFLICT (user_id) DO NOTHING;

-- Insert test credentials for users
-- SecretTestApiKey gets a private credential (API Key to access endpoints)
INSERT INTO credentials (credential_key, user_id, is_public, is_active) VALUES
    ('SecretTestApiKey', 'Docs User', false, true)
ON CONFLICT (credential_key) DO NOTHING;

-- Nillion2025 gets a private credential (API Key to access endpoints)
INSERT INTO credentials (credential_key, user_id, is_public, is_active) VALUES
    ('Nillion2025', 'Docs User', false, true)
ON CONFLICT (credential_key) DO NOTHING;

-- 030923f2e7120c50e42905b857ddd2947f6ecced6bb02aab64e63b28e9e2e06d10 gets a public credential (Public Keypair to access endpoints)
INSERT INTO credentials (credential_key, user_id, is_public, is_active) VALUES
    ('did:nil:030923f2e7120c50e42905b857ddd2947f6ecced6bb02aab64e63b28e9e2e06d10', 'Docs User', true, true)
ON CONFLICT (credential_key) DO NOTHING;
