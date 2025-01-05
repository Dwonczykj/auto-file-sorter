import sqlite3
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging


class GmailDatabase:
    def __init__(self, db_path: str = "auto_file_sorter/db/gmail.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self._ensure_db_directory()
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_db(self) -> None:
        """Initialize database tables if they don't exist"""
        cursor = self.conn.cursor()

        # Create gmail_labels table with uri column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gmail_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                uri TEXT,
                CHECK (name REGEXP '^[A-Za-z\s_\-]+$')
            )
        ''')

        # Create rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                rule_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create natural language rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS natural_language_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule TEXT NOT NULL CHECK(length(rule) <= 50),
                actions_json TEXT NOT NULL
            )
        ''')

        # Create blocked senders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocked_senders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL UNIQUE,
                type TEXT CHECK(type IN ('email', 'pattern', 'body_pattern')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Add REGEXP support for SQLite
        self.conn.create_function("REGEXP", 2, self._regexp)

        self.conn.commit()

    @staticmethod
    def _regexp(pattern: str, text: str) -> bool:
        """Custom REGEXP function for SQLite"""
        try:
            return bool(re.match(pattern, text))
        except Exception:
            return False

    # Label operations
    def create_label(self, name: str) -> Optional[int]:
        """Create a new label and return its ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'INSERT INTO gmail_labels (name) VALUES (?)', (name,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error creating label: {e}")
            return None

    def get_label(self, label_id: int) -> Optional[Dict[str, Any]]:
        """Get a label by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM gmail_labels WHERE id = ?', (label_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_labels(self) -> List[Dict[str, Any]]:
        """Get all labels"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM gmail_labels')
        return [dict(row) for row in cursor.fetchall()]

    def update_label(self, label_id: int, new_name: str) -> bool:
        """Update a label's name"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('UPDATE gmail_labels SET name = ? WHERE id = ?',
                           (new_name, label_id))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error updating label: {e}")
            return False

    def delete_label(self, label_id: int) -> bool:
        """Delete a label"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'DELETE FROM gmail_labels WHERE id = ?', (label_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error deleting label: {e}")
            return False

    def create_label_with_uri(self, name: str, uri: Optional[str] = None) -> Optional[int]:
        """Create a new label with URI and return its ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'INSERT INTO gmail_labels (name, uri) VALUES (?, ?)',
                (name, uri)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error creating label with URI: {e}")
            return None

    # Rule operations
    def create_rule(self, name: str, rule_data: Dict[str, Any]) -> Optional[int]:
        """Create a new rule and return its ID"""
        try:
            cursor = self.conn.cursor()
            rule_json = json.dumps(rule_data)
            cursor.execute(
                'INSERT INTO rules (name, rule_json) VALUES (?, ?)',
                (name, rule_json)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error creating rule: {e}")
            return None

    def get_rule(self, rule_id: int) -> Optional[Dict[str, Any]]:
        """Get a rule by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM rules WHERE id = ?', (rule_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['rule_json'] = json.loads(result['rule_json'])
            return result
        return None

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all rules"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM rules')
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['rule_json'] = json.loads(result['rule_json'])
            results.append(result)
        return results

    def update_rule(self, rule_id: int, name: str, rule_data: Dict[str, Any]) -> bool:
        """Update a rule"""
        try:
            cursor = self.conn.cursor()
            rule_json = json.dumps(rule_data)
            cursor.execute(
                'UPDATE rules SET name = ?, rule_json = ? WHERE id = ?',
                (name, rule_json, rule_id)
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error updating rule: {e}")
            return False

    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM rules WHERE id = ?', (rule_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error deleting rule: {e}")
            return False

    # Natural language rule operations
    def create_nl_rule(self, rule: str, actions: List[Dict[str, Any]]) -> Optional[int]:
        """Create a new natural language rule and return its ID"""
        try:
            cursor = self.conn.cursor()
            actions_json = json.dumps(actions)
            cursor.execute(
                'INSERT INTO natural_language_rules (rule, actions_json) VALUES (?, ?)',
                (rule, actions_json)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error creating natural language rule: {e}")
            return None

    def get_nl_rule(self, rule_id: int) -> Optional[Dict[str, Any]]:
        """Get a natural language rule by ID"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM natural_language_rules WHERE id = ?', (rule_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['actions'] = json.loads(result['actions_json'])
            return result
        return None

    def get_all_nl_rules(self) -> List[Dict[str, Any]]:
        """Get all natural language rules"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM natural_language_rules')
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['actions'] = json.loads(result['actions_json'])
            results.append(result)
        return results

    def delete_nl_rule(self, rule_id: int) -> bool:
        """Delete a natural language rule"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'DELETE FROM natural_language_rules WHERE id = ?', (rule_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error deleting natural language rule: {e}")
            return False

    def create_blocked_sender(self, pattern: str, pattern_type: str) -> Optional[int]:
        """
        Create a new blocked sender pattern
        pattern_type must be one of: 'email', 'pattern', 'body_pattern'
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'INSERT INTO blocked_senders (pattern, type) VALUES (?, ?)',
                (pattern, pattern_type)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error creating blocked sender: {e}")
            return None

    def get_all_blocked_senders(self) -> List[Dict[str, Any]]:
        """Get all blocked sender patterns"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM blocked_senders')
        return [dict(row) for row in cursor.fetchall()]

    def delete_blocked_sender(self, pattern: str) -> bool:
        """Delete a blocked sender pattern"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'DELETE FROM blocked_senders WHERE pattern = ?', (pattern,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error deleting blocked sender: {e}")
            return False

    def close(self) -> None:
        """Close the database connection"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
