#!/usr/bin/env python3
"""
轻量级知识图谱 - 专为E5Agent设计
结合NetworkX的图算法能力和SQLite的持久化存储
"""

import sqlite3
import json
import logging
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class FunctionNode:
    """轻量级函数节点"""
    name: str
    display_name: str
    description: str
    category: str
    domain: str
    keywords: List[str]
    complexity_score: float
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class FunctionRelation:
    """函数关系"""
    source: str
    target: str
    relation_type: str  # similar_to, depends_on, composes_with
    strength: float
    evidence_count: int = 1
    last_updated: str = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

class LightweightKnowledgeGraph:
    """轻量级知识图谱 - 无需外部数据库"""
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.db_path = self.memory_dir / "knowledge_graph.db"
        
        # NetworkX图用于算法计算
        self.graph = nx.DiGraph()
        self.similarity_graph = nx.Graph()  # 无向图用于相似性
        
        # 内存缓存
        self.nodes: Dict[str, FunctionNode] = {}
        self.relations: Dict[str, FunctionRelation] = {}
        
        # 语义分析
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.function_vectors = {}
        
        self._init_database()
        self._load_data()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.memory_dir.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS functions (
                    name TEXT PRIMARY KEY,
                    display_name TEXT,
                    description TEXT,
                    category TEXT,
                    domain TEXT,
                    keywords TEXT,  -- JSON array
                    complexity_score REAL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TEXT,
                    vector_embedding TEXT  -- JSON array
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    target TEXT,
                    relation_type TEXT,
                    strength REAL,
                    evidence_count INTEGER DEFAULT 1,
                    last_updated TEXT,
                    FOREIGN KEY (source) REFERENCES functions (name),
                    FOREIGN KEY (target) REFERENCES functions (name)
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON functions (category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON functions (domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_type ON relations (relation_type)")
    
    def add_function(self, name: str, description: str, complexity: Dict[str, Any]) -> bool:
        """添加函数到知识图谱"""
        try:
            # 分析函数特征
            category, domain = self._analyze_function_category(name, description)
            keywords = self._extract_keywords(name, description)
            display_name = self._generate_display_name(name)
            
            # 创建节点
            node = FunctionNode(
                name=name,
                display_name=display_name,
                description=description,
                category=category,
                domain=domain,
                keywords=keywords,
                complexity_score=complexity.get('complexity_score', 0)
            )
            
            # 计算语义向量
            vector = self._compute_semantic_vector(name, description)
            
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO functions 
                    (name, display_name, description, category, domain, keywords, 
                     complexity_score, usage_count, success_rate, created_at, vector_embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, display_name, description, category, domain,
                    json.dumps(keywords), node.complexity_score, node.usage_count,
                    node.success_rate, node.created_at, json.dumps(vector.tolist())
                ))
            
            # 更新内存缓存
            self.nodes[name] = node
            self.function_vectors[name] = vector
            self.graph.add_node(name, **asdict(node))
            
            # 分析与现有函数的关系
            self._analyze_relationships(name)
            
            logger.info(f"Added function to knowledge graph: {name} ({category}/{domain})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add function {name}: {e}")
            return False
    
    def _analyze_function_category(self, name: str, description: str) -> Tuple[str, str]:
        """分析函数类别和领域"""
        text = f"{name} {description}".lower()
        
        # 类别关键词
        categories = {
            'validation': ['validate', 'check', 'verify', 'test'],
            'calculation': ['calculate', 'compute', 'math', 'count'],
            'generation': ['create', 'generate', 'build', 'make'],
            'transformation': ['convert', 'transform', 'parse'],
            'analysis': ['analyze', 'process', 'examine']
        }
        
        # 领域关键词
        domains = {
            'web': ['html', 'css', 'website', 'url', 'http'],
            'security': ['password', 'encrypt', 'secure', 'auth'],
            'data': ['csv', 'json', 'data', 'file'],
            'math': ['calculator', 'math', 'number'],
            'text': ['string', 'text', 'word']
        }
        
        # 计算匹配分数
        best_category = max(categories.keys(), 
                          key=lambda c: sum(1 for k in categories[c] if k in text))
        best_domain = max(domains.keys(),
                        key=lambda d: sum(1 for k in domains[d] if k in text))
        
        return best_category, best_domain
    
    def _extract_keywords(self, name: str, description: str) -> List[str]:
        """提取关键词"""
        import re
        text = f"{name} {description}".lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # 过滤停用词
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'function'}
        keywords = [w for w in set(words) if w not in stop_words and len(w) > 2]
        
        return keywords[:8]  # 限制数量
    
    def _generate_display_name(self, name: str) -> str:
        """生成友好显示名"""
        # 移除时间戳
        clean_name = name
        if '_' in name and name.split('_')[-1].isdigit():
            clean_name = '_'.join(name.split('_')[:-1])
        
        return clean_name.replace('_', ' ').title()
    
    def _compute_semantic_vector(self, name: str, description: str) -> np.ndarray:
        """计算语义向量"""
        text = f"{name} {description}"

        # 收集所有文本用于训练向量化器
        all_texts = [text]
        for existing_name, existing_node in self.nodes.items():
            if existing_name != name:  # 避免重复
                all_texts.append(f"{existing_name} {existing_node.description}")

        # 重新训练向量化器
        if len(all_texts) > 1:
            self.vectorizer.fit(all_texts)
            vectors = self.vectorizer.transform(all_texts)

            # 更新所有现有函数的向量
            for i, (existing_name, _) in enumerate([(n, node) for n, node in self.nodes.items() if n != name]):
                self.function_vectors[existing_name] = vectors[i+1].toarray()[0]

            return vectors[0].toarray()[0]
        else:
            # 只有一个函数时
            self.vectorizer.fit([text])
            vector = self.vectorizer.transform([text])
            return vector.toarray()[0]
    
    def _analyze_relationships(self, new_function: str):
        """分析新函数与现有函数的关系"""
        if new_function not in self.nodes:
            return
        
        new_node = self.nodes[new_function]
        new_vector = self.function_vectors[new_function]
        
        for existing_name, existing_node in self.nodes.items():
            if existing_name == new_function:
                continue
            
            # 计算相似度
            similarity = self._calculate_similarity(new_node, existing_node, new_vector, existing_name)
            
            # 如果相似度足够高，建立关系
            if similarity > 0.3:
                self._add_relation(new_function, existing_name, 'similar_to', similarity)
    
    def _calculate_similarity(self, node1: FunctionNode, node2: FunctionNode, 
                            vector1: np.ndarray, func2_name: str) -> float:
        """计算函数相似度"""
        similarity = 0.0
        
        # 类别相似度
        if node1.category == node2.category:
            similarity += 0.3
        
        # 领域相似度
        if node1.domain == node2.domain:
            similarity += 0.2
        
        # 关键词相似度
        keywords1 = set(node1.keywords)
        keywords2 = set(node2.keywords)
        if keywords1 and keywords2:
            jaccard = len(keywords1 & keywords2) / len(keywords1 | keywords2)
            similarity += jaccard * 0.2
        
        # 语义向量相似度
        if func2_name in self.function_vectors:
            vector2 = self.function_vectors[func2_name]
            cosine_sim = cosine_similarity([vector1], [vector2])[0][0]
            similarity += cosine_sim * 0.3
        
        return similarity
    
    def _add_relation(self, source: str, target: str, rel_type: str, strength: float):
        """添加关系"""
        relation_id = f"{source}->{target}"
        
        relation = FunctionRelation(
            source=source,
            target=target,
            relation_type=rel_type,
            strength=strength
        )
        
        # 保存到数据库
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO relations 
                (id, source, target, relation_type, strength, evidence_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                relation_id, source, target, rel_type, strength,
                relation.evidence_count, relation.last_updated
            ))
        
        # 更新内存和图
        self.relations[relation_id] = relation
        self.graph.add_edge(source, target, **asdict(relation))
        self.similarity_graph.add_edge(source, target, weight=strength)
    
    def get_recommendations(self, task_description: str, limit: int = 5) -> List[Tuple[str, float, str]]:
        """获取函数推荐"""
        try:
            # 分析任务特征
            task_category, task_domain = self._analyze_function_category("", task_description)
            task_keywords = self._extract_keywords("", task_description)
            task_vector = self._compute_semantic_vector("", task_description)
            
            recommendations = []
            
            for func_name, node in self.nodes.items():
                score = 0.0
                reasons = []
                
                # 类别匹配
                if node.category == task_category:
                    score += 0.3
                    reasons.append(f"Category: {task_category}")
                
                # 领域匹配
                if node.domain == task_domain:
                    score += 0.2
                    reasons.append(f"Domain: {task_domain}")
                
                # 关键词匹配
                keyword_overlap = len(set(task_keywords) & set(node.keywords))
                if keyword_overlap > 0:
                    keyword_score = keyword_overlap / max(len(task_keywords), len(node.keywords))
                    score += keyword_score * 0.3
                    reasons.append(f"Keywords: {keyword_overlap}")
                
                # 语义相似度
                if func_name in self.function_vectors:
                    semantic_sim = cosine_similarity([task_vector], [self.function_vectors[func_name]])[0][0]
                    score += semantic_sim * 0.2
                    if semantic_sim > 0.1:
                        reasons.append(f"Semantic: {semantic_sim:.2f}")
                
                if score > 0.1:
                    recommendations.append((func_name, score, "; ".join(reasons)))
            
            # 排序并返回
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []
    
    def get_related_functions(self, func_name: str, limit: int = 5) -> List[Tuple[str, float, str]]:
        """获取相关函数"""
        if func_name not in self.graph:
            return []
        
        related = []
        
        # 直接关系
        for neighbor in self.graph.neighbors(func_name):
            edge_data = self.graph.get_edge_data(func_name, neighbor)
            strength = edge_data.get('strength', 0)
            rel_type = edge_data.get('relation_type', 'unknown')
            related.append((neighbor, strength, f"Direct: {rel_type}"))
        
        # 二度关系
        for neighbor in self.graph.neighbors(func_name):
            for second_neighbor in self.graph.neighbors(neighbor):
                if second_neighbor != func_name and second_neighbor not in [r[0] for r in related]:
                    # 计算间接关系强度
                    edge1 = self.graph.get_edge_data(func_name, neighbor)
                    edge2 = self.graph.get_edge_data(neighbor, second_neighbor)
                    indirect_strength = edge1.get('strength', 0) * edge2.get('strength', 0) * 0.5
                    if indirect_strength > 0.1:
                        related.append((second_neighbor, indirect_strength, f"Via: {neighbor}"))
        
        # 排序并返回
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:limit]
    
    def get_function_importance(self) -> Dict[str, float]:
        """计算函数重要性（PageRank）"""
        if not self.graph.nodes():
            return {}
        
        try:
            # 使用PageRank算法
            importance = nx.pagerank(self.graph, weight='strength')
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except:
            # 如果图为空或有问题，返回基于使用次数的重要性
            return {name: node.usage_count for name, node in self.nodes.items()}
    
    def find_function_communities(self) -> List[List[str]]:
        """发现函数社区"""
        if not self.similarity_graph.nodes():
            return []
        
        try:
            communities = nx.community.greedy_modularity_communities(self.similarity_graph)
            return [list(community) for community in communities if len(community) > 1]
        except:
            return []
    
    def _load_data(self):
        """从数据库加载数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 加载函数
                cursor = conn.execute("SELECT * FROM functions")
                for row in cursor.fetchall():
                    name, display_name, description, category, domain, keywords_json, \
                    complexity_score, usage_count, success_rate, created_at, vector_json = row
                    
                    node = FunctionNode(
                        name=name,
                        display_name=display_name,
                        description=description,
                        category=category,
                        domain=domain,
                        keywords=json.loads(keywords_json),
                        complexity_score=complexity_score,
                        usage_count=usage_count,
                        success_rate=success_rate,
                        created_at=created_at
                    )
                    
                    self.nodes[name] = node
                    self.graph.add_node(name, **asdict(node))
                    
                    if vector_json:
                        self.function_vectors[name] = np.array(json.loads(vector_json))
                
                # 加载关系
                cursor = conn.execute("SELECT * FROM relations")
                for row in cursor.fetchall():
                    relation_id, source, target, relation_type, strength, evidence_count, last_updated = row
                    
                    relation = FunctionRelation(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        strength=strength,
                        evidence_count=evidence_count,
                        last_updated=last_updated
                    )
                    
                    self.relations[relation_id] = relation
                    self.graph.add_edge(source, target, **asdict(relation))
                    if relation_type == 'similar_to':
                        self.similarity_graph.add_edge(source, target, weight=strength)
                
                logger.info(f"Loaded {len(self.nodes)} functions and {len(self.relations)} relations")
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_functions': len(self.nodes),
            'total_relations': len(self.relations),
            'categories': {cat: sum(1 for n in self.nodes.values() if n.category == cat) 
                         for cat in set(n.category for n in self.nodes.values())},
            'domains': {dom: sum(1 for n in self.nodes.values() if n.domain == dom)
                       for dom in set(n.domain for n in self.nodes.values())},
            'avg_complexity': sum(n.complexity_score for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            'most_important': list(self.get_function_importance().items())[:5]
        }
