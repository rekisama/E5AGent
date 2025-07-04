"""
Session Management and Token Usage Tracking for Agent System.

This module provides session context isolation, token usage tracking,
and rate limiting functionality for multi-agent conversations.
"""

import time
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class TokenUsageStats:
    """Track token usage and API call statistics."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_call_time: Optional[datetime] = None

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.api_calls += 1
        self.last_call_time = datetime.now()

    def get_rate_per_minute(self) -> float:
        """Calculate API calls per minute."""
        if self.api_calls == 0:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        return self.api_calls / max(elapsed, 1.0)

    def get_tokens_per_minute(self) -> float:
        """Calculate tokens per minute."""
        if self.total_tokens == 0:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        return self.total_tokens / max(elapsed, 1.0)

    def reset(self):
        """Reset all statistics."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        self.start_time = datetime.now()
        self.last_call_time = None


@dataclass
class SessionContext:
    """Isolated session context for multi-task scenarios."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsageStats = field(default_factory=TokenUsageStats)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clear_messages(self):
        """Clear session messages."""
        self.messages.clear()

    def add_message(self, message: Dict[str, Any]):
        """Add a message to the session."""
        message['timestamp'] = datetime.now()
        self.messages.append(message)

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)


class SessionManager:
    """
    Manages multiple session contexts with automatic cleanup and rate limiting.
    
    Features:
    - Session isolation for parallel tasks
    - Token usage tracking and rate limiting
    - Automatic cleanup of old sessions
    - Thread-safe operations
    """

    def __init__(self, max_tokens_per_minute: int = 10000, max_sessions: int = 100):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_sessions = max_sessions
        
        # Session storage
        self.sessions: Dict[str, SessionContext] = {}
        self.global_token_usage = TokenUsageStats()
        
        # Rate limiting
        self._last_api_call = 0.0
        self._min_interval = 1.0  # Minimum seconds between API calls
        self._lock = threading.Lock()

    def create_session(self, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session context.
        
        Args:
            session_id: Optional custom session ID
            metadata: Optional metadata to store with session
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        with self._lock:
            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_oldest_sessions(self.max_sessions // 2)
            
            session = SessionContext(
                session_id=session_id,
                metadata=metadata or {}
            )
            self.sessions[session_id] = session
            
        logger.debug(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session by ID."""
        with self._lock:
            return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Deleted session: {session_id}")
                return True
            return False

    def add_token_usage(self, session_id: str, prompt_tokens: int, completion_tokens: int):
        """Add token usage to both session and global tracking."""
        session = self.get_session(session_id)
        if session:
            session.token_usage.add_usage(prompt_tokens, completion_tokens)
        
        self.global_token_usage.add_usage(prompt_tokens, completion_tokens)

    def enforce_rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call

        if time_since_last_call < self._min_interval:
            sleep_time = self._min_interval - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_api_call = time.time()

        # Check token usage rate
        rate = self.global_token_usage.get_tokens_per_minute()
        if rate > self.max_tokens_per_minute:
            logger.warning(f"High token usage rate: {rate:.2f} tokens/min")

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session."""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            'session_id': session_id,
            'created_at': session.created_at,
            'message_count': session.get_message_count(),
            'token_usage': {
                'total_tokens': session.token_usage.total_tokens,
                'prompt_tokens': session.token_usage.prompt_tokens,
                'completion_tokens': session.token_usage.completion_tokens,
                'api_calls': session.token_usage.api_calls,
                'rate_per_minute': session.token_usage.get_rate_per_minute(),
                'tokens_per_minute': session.token_usage.get_tokens_per_minute()
            },
            'metadata': session.metadata
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all sessions."""
        with self._lock:
            active_sessions = len(self.sessions)
            
        return {
            'active_sessions': active_sessions,
            'global_token_usage': {
                'total_tokens': self.global_token_usage.total_tokens,
                'prompt_tokens': self.global_token_usage.prompt_tokens,
                'completion_tokens': self.global_token_usage.completion_tokens,
                'api_calls': self.global_token_usage.api_calls,
                'rate_per_minute': self.global_token_usage.get_rate_per_minute(),
                'tokens_per_minute': self.global_token_usage.get_tokens_per_minute()
            }
        }

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old sessions to prevent memory leaks."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            old_sessions = [
                sid for sid, session in self.sessions.items()
                if session.created_at < cutoff_time
            ]

            for sid in old_sessions:
                del self.sessions[sid]

        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old sessions")
            
        return len(old_sessions)

    def _cleanup_oldest_sessions(self, count: int):
        """Clean up the oldest sessions (internal method)."""
        if not self.sessions:
            return
            
        # Sort sessions by creation time
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove oldest sessions
        for i in range(min(count, len(sorted_sessions))):
            session_id = sorted_sessions[i][0]
            del self.sessions[session_id]
            logger.debug(f"Cleaned up old session: {session_id}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with basic info."""
        with self._lock:
            sessions_info = []
            for session_id, session in self.sessions.items():
                sessions_info.append({
                    'session_id': session_id,
                    'created_at': session.created_at,
                    'message_count': session.get_message_count(),
                    'total_tokens': session.token_usage.total_tokens,
                    'metadata': session.metadata
                })
            
        return sorted(sessions_info, key=lambda x: x['created_at'], reverse=True)


# Global session manager instance
_global_session_manager: Optional[SessionManager] = None


def get_session_manager(max_tokens_per_minute: int = 10000) -> SessionManager:
    """Get or create global session manager instance."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager(max_tokens_per_minute=max_tokens_per_minute)
    return _global_session_manager


def reset_session_manager():
    """Reset global session manager (mainly for testing)."""
    global _global_session_manager
    _global_session_manager = None
