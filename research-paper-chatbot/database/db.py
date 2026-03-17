"""
PostgreSQL Database Connection Module
Provides connection pooling and database utilities for the Flask application.
"""
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import os
from contextlib import contextmanager
from dotenv import load_dotenv   # ✅ ADD THIS

load_dotenv()  # ✅ ADD THIS
# Database configuration - read from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'paperiq_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
}
print("DB HOST:", os.getenv("DB_HOST"))
# Connection pool - initialized when the module is loaded
connection_pool = None


def init_db_pool(minconn=1, maxconn=10):
    """
    Initialize the PostgreSQL connection pool.
    Should be called once when the application starts.
    
    Args:
        minconn (int): Minimum number of connections to maintain
        maxconn (int): Maximum number of connections allowed
    """
    global connection_pool
    try:
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn,
            maxconn,
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        print(f"✓ Database connection pool initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing database pool: {e}")
        return False


@contextmanager
def get_db_connection():
    """
    Context manager for getting a database connection from the pool.
    Automatically returns the connection to the pool after use.
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            results = cursor.fetchall()
    """
    conn = None
    try:
        if connection_pool is None:
            raise Exception("Database pool not initialized. Call init_db_pool() first.")
        
        conn = connection_pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            connection_pool.putconn(conn)


@contextmanager
def get_db_cursor(cursor_factory=RealDictCursor):
    """
    Context manager for getting a database cursor.
    Returns results as dictionaries by default for easier JSON serialization.
    Automatically handles connection and cursor cleanup.
    
    Args:
        cursor_factory: Type of cursor to use (default: RealDictCursor for dict results)
    
    Usage:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()


def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """
    Execute a database query and return results.
    
    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters for safe SQL execution
        fetch_one (bool): Return single result
        fetch_all (bool): Return all results
    
    Returns:
        Query results as dict(s) or None for INSERT/UPDATE/DELETE
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(query, params)
            
            if fetch_one:
                return cursor.fetchone()
            elif fetch_all:
                return cursor.fetchall()
            else:
                return None
    except Exception as e:
        print(f"Database query error: {e}")
        raise e


def close_db_pool():
    """
    Close all connections in the pool.
    Should be called when the application shuts down.
    """
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        print("✓ Database connection pool closed")


# Test database connection
def test_connection():
    """
    Test the database connection by executing a simple query.
    Returns True if successful, False otherwise.
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result is not None
    except Exception as e:
        print(f"✗ Database connection test failed: {e}")
        return False
