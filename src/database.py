"""
Database module for managing conversation history and context information.
"""
import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import aiosqlite
import json
from loguru import logger


class DatabaseManager:
    """Manages SQLite database operations for the voice AI agent."""
    
    def __init__(self, db_path: str = "data/voice_agent.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def init_database(self):
        """Initialize database with required tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # Conversations table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    audio_file_path TEXT,
                    transcribed_text TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    processing_time REAL,
                    model_used TEXT DEFAULT 'base',
                    confidence_score REAL,
                    metadata TEXT  -- JSON string for additional info
                )
            """)
            
            # Context information table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS context_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # User preferences table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key TEXT NOT NULL UNIQUE,
                    setting_value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Audio files metadata
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    original_filename TEXT,
                    file_size INTEGER,
                    duration REAL,
                    format TEXT,
                    sample_rate INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
            logger.info("Database initialized successfully")
    
    async def save_conversation(
        self, 
        transcribed_text: str,
        ai_response: str,
        audio_file_path: Optional[str] = None,
        processing_time: Optional[float] = None,
        model_used: str = "base",
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save a conversation to the database."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO conversations 
                (audio_file_path, transcribed_text, ai_response, processing_time, 
                 model_used, confidence_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                audio_file_path,
                transcribed_text,
                ai_response,
                processing_time,
                model_used,
                confidence_score,
                json.dumps(metadata) if metadata else None
            ))
            await db.commit()
            conversation_id = cursor.lastrowid
            logger.info(f"Saved conversation with ID: {conversation_id}")
            return conversation_id
    
    async def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent conversations."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_conversation_by_id(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific conversation by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM conversations WHERE id = ?
            """, (conversation_id,)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
    
    async def save_context_info(self, key: str, value: str, category: str = "general"):
        """Save context information."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO context_info (key, value, category, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (key, value, category))
            await db.commit()
            logger.info(f"Saved context info: {key}")
    
    async def get_context_info(self, key: str) -> Optional[str]:
        """Retrieve context information and update access stats."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get the value
            async with db.execute("""
                SELECT value FROM context_info WHERE key = ?
            """, (key,)) as cursor:
                row = await cursor.fetchone()
                
            if row:
                # Update access stats
                await db.execute("""
                    UPDATE context_info 
                    SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE key = ?
                """, (key,))
                await db.commit()
                return row[0]
            return None
    
    async def get_context_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all context information by category."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM context_info WHERE category = ?
                ORDER BY last_accessed DESC
            """, (category,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def save_audio_metadata(
        self,
        file_path: str,
        original_filename: str,
        file_size: int,
        duration: float,
        format: str,
        sample_rate: int
    ) -> int:
        """Save audio file metadata."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO audio_files 
                (file_path, original_filename, file_size, duration, format, sample_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, original_filename, file_size, duration, format, sample_rate))
            await db.commit()
            return cursor.lastrowid
    
    async def search_conversations(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversations by text content."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            search_query = f"%{query}%"
            async with db.execute("""
                SELECT * FROM conversations 
                WHERE transcribed_text LIKE ? OR ai_response LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (search_query, search_query, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}
            
            # Total conversations
            async with db.execute("SELECT COUNT(*) FROM conversations") as cursor:
                stats['total_conversations'] = (await cursor.fetchone())[0]
            
            # Average processing time
            async with db.execute("""
                SELECT AVG(processing_time) FROM conversations 
                WHERE processing_time IS NOT NULL
            """) as cursor:
                avg_time = (await cursor.fetchone())[0]
                stats['avg_processing_time'] = round(avg_time, 2) if avg_time else 0
            
            # Most used model
            async with db.execute("""
                SELECT model_used, COUNT(*) as count FROM conversations 
                GROUP BY model_used ORDER BY count DESC LIMIT 1
            """) as cursor:
                row = await cursor.fetchone()
                stats['most_used_model'] = row[0] if row else "None"
            
            # Context info count
            async with db.execute("SELECT COUNT(*) FROM context_info") as cursor:
                stats['context_entries'] = (await cursor.fetchone())[0]
            
            return stats
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data beyond specified days."""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete old conversations
            await db.execute("""
                DELETE FROM conversations 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            # Delete unused context info
            await db.execute("""
                DELETE FROM context_info 
                WHERE last_accessed < datetime('now', '-{} days')
                AND access_count < 5
            """.format(days))
            
            await db.commit()
            logger.info(f"Cleaned up data older than {days} days")

# Global database instance
db_manager = DatabaseManager()