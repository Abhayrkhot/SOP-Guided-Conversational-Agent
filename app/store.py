"""SQLite transactions serialize each turn, including the demo email side effect."""
import sqlite3
import time
from contextlib import contextmanager
from threading import RLock
from .models import Session

class Store:
    def __init__(self, path=':memory:', ttl=1800):
        self.ttl = ttl
        self.lock = RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=15)
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL, expires REAL NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS outbox (session_id TEXT PRIMARY KEY, recipient TEXT, subject TEXT, body TEXT, expires REAL NOT NULL)')
        self.db.commit()

    @contextmanager
    def turn(self):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                now = time.time()
                self.db.execute('DELETE FROM sessions WHERE expires < ?', (now,))
                self.db.execute('DELETE FROM outbox WHERE expires < ?', (now,))
                yield
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def get(self, sid):
        row = self.db.execute('SELECT state FROM sessions WHERE id=? AND expires>=?', (sid, time.time())).fetchone()
        return Session.model_validate_json(row[0]) if row else None

    def save(self, sid, state):
        self.db.execute('INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)',
                        (sid, state.model_dump_json(), time.time() + self.ttl))

    def queue(self, sid, recipient, body):
        self.db.execute('INSERT OR IGNORE INTO outbox VALUES (?, ?, ?, ?, ?)',
                        (sid, recipient, 'Insurance claim conversation summary', body, time.time() + self.ttl))

    def messages(self):
        with self.lock:
            return [dict(zip(('to', 'subject', 'body'), row)) for row in self.db.execute(
                'SELECT recipient, subject, body FROM outbox WHERE expires>=?', (time.time(),))]
