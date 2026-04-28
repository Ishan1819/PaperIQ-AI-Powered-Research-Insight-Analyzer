"""
Database initialization script.
Creates the required tables for the authentication system.
Run this script once to set up the database schema.
"""

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'paperiq_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

def create_tables():
    """Create users and chat_history tables if they don't exist"""
    
    # SQL statements to create tables
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_chat_history_table = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        user_message TEXT,
        ai_response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create indexes for better performance
    create_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);"
    ]
    
    try:
        # Connect to PostgreSQL
        print(f"Connecting to database: {DB_CONFIG['database']} at {DB_CONFIG['host']}...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("Creating users table...")
        cursor.execute(create_users_table)
        
        print("Creating chat_history table...")
        cursor.execute(create_chat_history_table)
        
        print("Creating indexes...")
        for index_sql in create_indexes:
            cursor.execute(index_sql)
        
        # Commit changes
        conn.commit()
        
        print("\n✅ Database tables created successfully!")
        print("\nTables created:")
        print("  - users (id, email, password_hash, created_at)")
        print("  - chat_history (id, user_id, user_message, ai_response, created_at)")
        print("\nIndexes created:")
        print("  - idx_chat_history_user_id")
        print("  - idx_chat_history_created_at")
        print("  - idx_users_email")
        
        # Close connection
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_connection():
    """Test database connection"""
    try:
        print(f"\nTesting connection to {DB_CONFIG['database']}...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connected successfully!")
        print(f"PostgreSQL version: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PaperIQ Database Initialization")
    print("=" * 60)
    
    # Test connection first
    if test_connection():
        print("\n" + "=" * 60)
        print("Creating database tables...")
        print("=" * 60)
        create_tables()
    else:
        print("\n⚠️  Please check your database configuration in .env file")
        print("\nRequired environment variables:")
        print("  - DB_HOST")
        print("  - DB_PORT")
        print("  - DB_NAME")
        print("  - DB_USER")
        print("  - DB_PASSWORD")
