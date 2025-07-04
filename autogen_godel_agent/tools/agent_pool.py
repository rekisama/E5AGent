"""
Agent Pool Management for AutoGen Agents.

This module provides efficient pooling and reuse of UserProxyAgent instances
to avoid the overhead of creating new agents for each conversation.
"""

import threading
import uuid
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
import logging
import autogen

# Configure logger
logger = logging.getLogger(__name__)


class UserProxyPool:
    """
    Pool of UserProxyAgent instances for reuse.
    
    Features:
    - Thread-safe agent pooling
    - Automatic cleanup of agent state
    - Configurable pool size limits
    - Context manager support for safe usage
    """

    def __init__(self, max_size: int = 5, agent_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the UserProxy pool.
        
        Args:
            max_size: Maximum number of agents to keep in pool
            agent_config: Default configuration for creating new agents
        """
        self.max_size = max_size
        self.agent_config = agent_config or {
            "human_input_mode": "NEVER",
            "max_consecutive_auto_reply": 1,
            "code_execution_config": False,
        }
        
        self._pool: List[autogen.UserProxyAgent] = []
        self._lock = threading.Lock()
        self._created_count = 0
        self._reused_count = 0

    @contextmanager
    def get_user_proxy(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Get a UserProxyAgent from the pool as a context manager.
        
        Args:
            custom_config: Optional custom configuration for this agent
            
        Yields:
            UserProxyAgent instance
        """
        proxy = None
        try:
            with self._lock:
                if self._pool:
                    proxy = self._pool.pop()
                    self._reused_count += 1
                    logger.debug(f"Reused agent from pool (pool size: {len(self._pool)})")
                else:
                    # Create new agent with merged configuration
                    config = self.agent_config.copy()
                    if custom_config:
                        config.update(custom_config)
                    
                    proxy = autogen.UserProxyAgent(
                        name=f"temp_user_{uuid.uuid4().hex[:8]}",
                        **config
                    )
                    self._created_count += 1
                    logger.debug(f"Created new agent (total created: {self._created_count})")

            # Clear any existing messages and state
            self._clean_agent_state(proxy)
            yield proxy

        finally:
            if proxy:
                # Clean up and return to pool
                self._clean_agent_state(proxy)
                with self._lock:
                    if len(self._pool) < self.max_size:
                        self._pool.append(proxy)
                        logger.debug(f"Returned agent to pool (pool size: {len(self._pool)})")
                    else:
                        logger.debug("Pool full, discarding agent")

    def _clean_agent_state(self, agent: autogen.UserProxyAgent):
        """Clean agent state for reuse."""
        try:
            # Clear chat messages
            if hasattr(agent, 'chat_messages'):
                agent.chat_messages.clear()
            
            # Reset any other stateful attributes if needed
            if hasattr(agent, '_oai_messages'):
                agent._oai_messages.clear()
                
            # Reset consecutive auto reply count
            if hasattr(agent, '_consecutive_auto_reply_counter'):
                agent._consecutive_auto_reply_counter = {}
                
        except Exception as e:
            logger.warning(f"Error cleaning agent state: {e}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'created_count': self._created_count,
                'reused_count': self._reused_count,
                'reuse_ratio': self._reused_count / max(self._created_count, 1)
            }

    def clear_pool(self):
        """Clear all agents from the pool."""
        with self._lock:
            self._pool.clear()
            logger.info("Cleared agent pool")

    def resize_pool(self, new_max_size: int):
        """Resize the pool maximum size."""
        with self._lock:
            old_size = self.max_size
            self.max_size = new_max_size
            
            # If reducing size, remove excess agents
            if new_max_size < len(self._pool):
                excess = len(self._pool) - new_max_size
                for _ in range(excess):
                    self._pool.pop()
                    
            logger.info(f"Resized pool from {old_size} to {new_max_size}")


class AssistantAgentPool:
    """
    Pool for AssistantAgent instances with different configurations.
    
    This is useful when you need multiple assistant agents with different
    system messages or configurations.
    """

    def __init__(self, max_size: int = 3):
        """
        Initialize the AssistantAgent pool.
        
        Args:
            max_size: Maximum number of agents to keep in pool per configuration
        """
        self.max_size = max_size
        self._pools: Dict[str, List[autogen.AssistantAgent]] = {}
        self._lock = threading.Lock()
        self._created_count = 0
        self._reused_count = 0

    @contextmanager
    def get_assistant_agent(self, config_key: str, agent_config: Dict[str, Any]):
        """
        Get an AssistantAgent from the pool as a context manager.
        
        Args:
            config_key: Unique key identifying the agent configuration
            agent_config: Configuration for creating the agent
            
        Yields:
            AssistantAgent instance
        """
        agent = None
        try:
            with self._lock:
                if config_key in self._pools and self._pools[config_key]:
                    agent = self._pools[config_key].pop()
                    self._reused_count += 1
                    logger.debug(f"Reused assistant agent for config: {config_key}")
                else:
                    # Create new agent
                    agent = autogen.AssistantAgent(**agent_config)
                    self._created_count += 1
                    logger.debug(f"Created new assistant agent for config: {config_key}")

            # Clear any existing state
            self._clean_agent_state(agent)
            yield agent

        finally:
            if agent:
                # Clean up and return to pool
                self._clean_agent_state(agent)
                with self._lock:
                    if config_key not in self._pools:
                        self._pools[config_key] = []
                    
                    if len(self._pools[config_key]) < self.max_size:
                        self._pools[config_key].append(agent)
                        logger.debug(f"Returned assistant agent to pool: {config_key}")

    def _clean_agent_state(self, agent: autogen.AssistantAgent):
        """Clean agent state for reuse."""
        try:
            # Clear chat messages
            if hasattr(agent, 'chat_messages'):
                agent.chat_messages.clear()
            
            # Reset any other stateful attributes
            if hasattr(agent, '_oai_messages'):
                agent._oai_messages.clear()
                
        except Exception as e:
            logger.warning(f"Error cleaning assistant agent state: {e}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_agents = sum(len(pool) for pool in self._pools.values())
            return {
                'total_agents': total_agents,
                'pools': {key: len(pool) for key, pool in self._pools.items()},
                'max_size_per_pool': self.max_size,
                'created_count': self._created_count,
                'reused_count': self._reused_count,
                'reuse_ratio': self._reused_count / max(self._created_count, 1)
            }

    def clear_pools(self):
        """Clear all agent pools."""
        with self._lock:
            self._pools.clear()
            logger.info("Cleared all assistant agent pools")


# Global pool instances
_global_user_proxy_pool: Optional[UserProxyPool] = None
_global_assistant_pool: Optional[AssistantAgentPool] = None


def get_user_proxy_pool(max_size: int = 5, agent_config: Optional[Dict[str, Any]] = None) -> UserProxyPool:
    """Get or create global UserProxy pool instance."""
    global _global_user_proxy_pool
    if _global_user_proxy_pool is None:
        _global_user_proxy_pool = UserProxyPool(max_size=max_size, agent_config=agent_config)
    return _global_user_proxy_pool


def get_assistant_agent_pool(max_size: int = 3) -> AssistantAgentPool:
    """Get or create global AssistantAgent pool instance."""
    global _global_assistant_pool
    if _global_assistant_pool is None:
        _global_assistant_pool = AssistantAgentPool(max_size=max_size)
    return _global_assistant_pool


def reset_agent_pools():
    """Reset all global agent pools (mainly for testing)."""
    global _global_user_proxy_pool, _global_assistant_pool
    if _global_user_proxy_pool:
        _global_user_proxy_pool.clear_pool()
    if _global_assistant_pool:
        _global_assistant_pool.clear_pools()
    _global_user_proxy_pool = None
    _global_assistant_pool = None
