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
-- Nillion2025 gets a private credential (API Key to access endpoints)
INSERT INTO credentials (credential_key, user_id, is_public, is_active) VALUES
    ('Nillion2025', 'Docs User', false, true)
ON CONFLICT (credential_key) DO NOTHING;

-- abc-def-ghi-123 gets a public credential (Public Keypair to access endpoints)
INSERT INTO credentials (credential_key, user_id, is_public, is_active) VALUES
    ('abc_private_key_123', 'Docs User', true, true)
ON CONFLICT (credential_key) DO NOTHING;
