-- scripts/init.sql
-- Database initialization script for Agentic AI Agent Platform

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create enums
DO $$ BEGIN
    CREATE TYPE campaign_status AS ENUM (
        'draft', 'pending_approval', 'approved', 
        'running', 'paused', 'completed', 'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE content_status AS ENUM (
        'generated', 'pending_review', 'approved', 
        'rejected', 'deployed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE platform_enum AS ENUM (
        'linkedin', 'twitter', 'email'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create delayed_rewards table
CREATE TABLE IF NOT EXISTS delayed_rewards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL,
    lead_email VARCHAR(255) NOT NULL,
    lead_data JSONB,
    initial_reward FLOAT DEFAULT 1.0,
    current_reward FLOAT DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'pending',
    booking_data JSONB,
    registered_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create scraped_content table
CREATE TABLE IF NOT EXISTS scraped_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    keywords TEXT,
    insights JSONB,
    scraped_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_delayed_rewards_status ON delayed_rewards(status);
CREATE INDEX IF NOT EXISTS idx_delayed_rewards_campaign ON delayed_rewards(campaign_id);
CREATE INDEX IF NOT EXISTS idx_delayed_rewards_email ON delayed_rewards(lead_email);
CREATE INDEX IF NOT EXISTS idx_scraped_content_source ON scraped_content(source);
CREATE INDEX IF NOT EXISTS idx_scraped_content_scraped_at ON scraped_content(scraped_at);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agentic;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agentic;