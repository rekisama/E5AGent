"""
GUI Workflow Visualizer

This module provides graphical user interface for workflow visualization
using tkinter and matplotlib to create interactive popup windows.

Features:
- Interactive workflow diagrams
- Real-time node status updates
- Zoom and pan capabilities
- Export to image files
- Multiple layout algorithms
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Try to import GUI libraries
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import networkx as nx

    # Configure matplotlib for Chinese font support
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    GUI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GUI libraries not available: {e}")
    GUI_AVAILABLE = False


class WorkflowGUIVisualizer:
    """
    GUI-based workflow visualizer with interactive popup windows.
    """
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("GUI libraries (tkinter, matplotlib, networkx) are required for GUI visualization")
        
        self.root = None
        self.current_workflow = None
        self.figure = None
        self.canvas = None
        self.graph = None
        
        logger.info("✅ GUI Workflow Visualizer initialized")
    
    def show_workflow_popup(self, task_description: str, workflow_type: str = "standard",
                           workflow_data: Dict = None) -> None:
        """
        Show workflow in a popup window.

        Args:
            task_description: Description of the task
            workflow_type: Type of workflow (standard, evo, multifile)
            workflow_data: Optional workflow data structure
        """

        try:
            # Create and show GUI directly in main thread
            self._create_main_window(task_description, workflow_type, workflow_data)
            logger.info(f"🖼️ Opened workflow visualization window for: {task_description}")

            # Bring window to front
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))

            # Start the GUI event loop
            self.root.mainloop()

        except Exception as e:
            logger.error(f"GUI error: {e}")
            if GUI_AVAILABLE:
                messagebox.showerror("Error", f"Failed to create GUI: {e}")
            print(f"❌ GUI Error: {e}")
    
    def _create_main_window(self, task_description: str, workflow_type: str, workflow_data: Dict):
        """Create the main visualization window."""
        
        self.root = tk.Tk()
        self.root.title(f"AutoGen Workflow Visualizer - {workflow_type.upper()}")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(title_frame, text=f"工作流可视化: {task_description}",
                               font=('Microsoft YaHei', 14, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # Create control buttons
        button_frame = ttk.Frame(title_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="刷新", command=lambda: self._refresh_visualization(task_description, workflow_type)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="导出PNG", command=self._export_png).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="导出SVG", command=self._export_svg).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="关闭", command=self.root.destroy).pack(side=tk.LEFT, padx=2)
        
        # Create notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Graph view tab
        graph_frame = ttk.Frame(notebook)
        notebook.add(graph_frame, text="流程图")
        self._create_graph_view(graph_frame, task_description, workflow_type)

        # Details view tab
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="详细信息")
        self._create_details_view(details_frame, task_description, workflow_type, workflow_data)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text=f"工作流类型: {workflow_type} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.status_label.pack(side=tk.LEFT)
    
    def _create_graph_view(self, parent: ttk.Frame, task_description: str, workflow_type: str):
        """Create the graph visualization view."""
        
        # Create matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.figure.patch.set_facecolor('white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        
        # Generate and display workflow graph
        self._generate_workflow_graph(task_description, workflow_type)
    
    def _generate_workflow_graph(self, task_description: str, workflow_type: str):
        """Generate and display the workflow graph."""
        
        self.ax.clear()
        self.ax.set_title(f"工作流图: {task_description}", fontsize=16, fontweight='bold', pad=20)
        
        # Create NetworkX graph
        self.graph = nx.DiGraph()
        
        if workflow_type == "standard":
            self._create_standard_graph()
        elif workflow_type == "evo":
            self._create_evo_graph()
        elif workflow_type == "multifile":
            self._create_multifile_graph()
        else:
            self._create_default_graph()
        
        # Layout the graph
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_colors.append(node_data.get('color', '#lightblue'))
            node_sizes.append(node_data.get('size', 2000))
        
        nx.draw_networkx_nodes(self.graph, pos, ax=self.ax, 
                              node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, ax=self.ax, 
                              edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        labels = {node: node for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, ax=self.ax, font_size=8, font_weight='bold')
        
        self.ax.set_axis_off()
        self.canvas.draw()
    
    def _create_standard_graph(self):
        """Create graph for standard workflow."""
        nodes = [
            ("Task Input", {"color": "#e3f2fd", "size": 2000}),
            ("Task Planning", {"color": "#f3e5f5", "size": 2000}),
            ("Search Functions", {"color": "#f3e5f5", "size": 2000}),
            ("Function Match?", {"color": "#fff3e0", "size": 1500}),
            ("Execute Existing", {"color": "#e8f5e8", "size": 2000}),
            ("Create New Function", {"color": "#f3e5f5", "size": 2000}),
            ("Security Validation", {"color": "#f3e5f5", "size": 1800}),
            ("Generate Tests", {"color": "#f3e5f5", "size": 1800}),
            ("Register Function", {"color": "#f3e5f5", "size": 1800}),
            ("Return Result", {"color": "#e8f5e8", "size": 2000})
        ]

        edges = [
            ("Task Input", "Task Planning"),
            ("Task Planning", "Search Functions"),
            ("Search Functions", "Function Match?"),
            ("Function Match?", "Execute Existing"),
            ("Function Match?", "Create New Function"),
            ("Create New Function", "Security Validation"),
            ("Security Validation", "Generate Tests"),
            ("Generate Tests", "Register Function"),
            ("Register Function", "Return Result"),
            ("Execute Existing", "Return Result")
        ]
        
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
    
    def _create_evo_graph(self):
        """Create graph for EvoWorkflow."""
        nodes = [
            ("复杂任务", {"color": "#e3f2fd", "size": 2500}),
            ("任务规划器", {"color": "#f3e5f5", "size": 2000}),
            ("子任务分解", {"color": "#f3e5f5", "size": 2000}),
            ("依赖分析", {"color": "#f3e5f5", "size": 2000}),
            ("工作流图", {"color": "#fff3e0", "size": 2000}),
            ("代理配置", {"color": "#f3e5f5", "size": 2000}),
            ("分析节点", {"color": "#e8f5e8", "size": 1800}),
            ("执行节点", {"color": "#e8f5e8", "size": 1800}),
            ("验证节点", {"color": "#e8f5e8", "size": 1800}),
            ("聚合节点", {"color": "#e8f5e8", "size": 1800}),
            ("最终结果", {"color": "#e8f5e8", "size": 2500})
        ]
        
        edges = [
            ("复杂任务", "任务规划器"),
            ("任务规划器", "子任务分解"),
            ("子任务分解", "依赖分析"),
            ("依赖分析", "工作流图"),
            ("工作流图", "代理配置"),
            ("代理配置", "分析节点"),
            ("代理配置", "执行节点"),
            ("代理配置", "验证节点"),
            ("分析节点", "执行节点"),
            ("执行节点", "验证节点"),
            ("验证节点", "聚合节点"),
            ("聚合节点", "最终结果")
        ]
        
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
    
    def _create_multifile_graph(self):
        """Create graph for MultiFile workflow."""
        nodes = [
            ("项目需求", {"color": "#e3f2fd", "size": 2500}),
            ("类型分类", {"color": "#f3e5f5", "size": 2000}),
            ("项目规划", {"color": "#f3e5f5", "size": 2000}),
            ("目录创建", {"color": "#fff3e0", "size": 1800}),
            ("HTML生成", {"color": "#e8f5e8", "size": 1800}),
            ("CSS生成", {"color": "#e8f5e8", "size": 1800}),
            ("JS生成", {"color": "#e8f5e8", "size": 1800}),
            ("README生成", {"color": "#e8f5e8", "size": 1800}),
            ("文件写入", {"color": "#fff3e0", "size": 2000}),
            ("项目完成", {"color": "#e8f5e8", "size": 2500})
        ]
        
        edges = [
            ("项目需求", "类型分类"),
            ("类型分类", "项目规划"),
            ("项目规划", "目录创建"),
            ("目录创建", "HTML生成"),
            ("目录创建", "CSS生成"),
            ("目录创建", "JS生成"),
            ("目录创建", "README生成"),
            ("HTML生成", "文件写入"),
            ("CSS生成", "文件写入"),
            ("JS生成", "文件写入"),
            ("README生成", "文件写入"),
            ("文件写入", "项目完成")
        ]
        
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
    
    def _create_default_graph(self):
        """Create default graph."""
        nodes = [
            ("开始", {"color": "#e3f2fd", "size": 2000}),
            ("处理", {"color": "#f3e5f5", "size": 2000}),
            ("结束", {"color": "#e8f5e8", "size": 2000})
        ]
        
        edges = [("开始", "处理"), ("处理", "结束")]
        
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
    
    def _create_details_view(self, parent: ttk.Frame, task_description: str, 
                           workflow_type: str, workflow_data: Dict):
        """Create the details view."""
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate details content
        details = self._generate_details_text(task_description, workflow_type, workflow_data)
        text_widget.insert(tk.END, details)
        text_widget.config(state=tk.DISABLED)
    
    def _generate_details_text(self, task_description: str, workflow_type: str, 
                              workflow_data: Dict) -> str:
        """Generate detailed text description."""
        
        details = f"""
工作流详细信息
{'='*60}

任务描述: {task_description}
工作流类型: {workflow_type.upper()}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}

"""
        
        if workflow_type == "standard":
            details += """
标准任务处理流程:

1. 任务规划代理
   - 分析用户任务需求
   - 理解任务语义和目标
   - 确定处理策略

2. 函数搜索
   - 在函数注册表中搜索匹配函数
   - 使用语义匹配算法
   - 评估函数相关性

3. 决策分支
   - 如果找到合适函数 → 直接执行
   - 如果未找到 → 创建新函数

4. 函数创建流程 (如需要)
   - 生成函数代码
   - AST安全验证
   - 生成测试用例
   - 执行测试验证
   - 注册到函数库

5. 执行与结果
   - 执行选定的函数
   - 记录执行历史
   - 返回处理结果
"""
        
        elif workflow_type == "evo":
            details += """
EvoWorkflow 复杂任务处理:

阶段1: 工作流生成
- 任务规划器使用LLM分析复杂任务
- 将任务分解为可执行的子任务
- 分析子任务间的依赖关系
- 创建有向无环图(DAG)工作流

阶段2: 代理配置
- 为每个工作流节点创建专门的AutoGen代理
- 配置代理角色: 分析器、执行器、验证器、聚合器
- 建立代理间的通信机制

阶段3: 执行引擎
- 使用拓扑排序确定最优执行顺序
- 支持并行执行独立节点
- 实时监控执行状态
- 错误处理和恢复机制
- 结果验证和质量保证

阶段4: 结果聚合
- 收集所有节点的执行结果
- 执行最终的聚合逻辑
- 生成综合性报告
- 更新性能指标
"""
        
        elif workflow_type == "multifile":
            details += """
MultiFile 项目生成流程:

步骤1: 项目分析
- 任务类型分类器识别项目类型
  * 网页项目 (HTML/CSS/JavaScript)
  * API项目 (Python/FastAPI)
  * 数据分析 (Jupyter/Pandas)
  * 文档项目 (Markdown/静态站点)

步骤2: 项目规划
- 根据项目类型选择模板
- 确定目录结构
- 规划文件列表和依赖关系

步骤3: 内容生成
- 使用LLM为每个文件生成内容
- 确保文件间的一致性
- 遵循最佳实践和编码规范

步骤4: 项目构建
- 创建目录结构
- 写入所有生成的文件
- 生成配置文件 (package.json, requirements.txt等)
- 创建项目文档

步骤5: 质量保证
- 验证文件完整性
- 检查语法错误
- 生成使用说明
"""
        
        if workflow_data:
            details += f"\n\n原始工作流数据:\n{'-'*30}\n"
            details += json.dumps(workflow_data, indent=2, ensure_ascii=False)
        
        return details
    
    def _refresh_visualization(self, task_description: str, workflow_type: str):
        """Refresh the visualization."""
        self._generate_workflow_graph(task_description, workflow_type)
        self.status_label.config(text=f"工作流类型: {workflow_type} | 刷新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _export_png(self):
        """Export graph as PNG."""
        if self.figure:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("导出成功", f"图表已保存到: {filename}")
    
    def _export_svg(self):
        """Export graph as SVG."""
        if self.figure:
            filename = filedialog.asksaveasfilename(
                defaultextension=".svg",
                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
            )
            if filename:
                self.figure.savefig(filename, format='svg', bbox_inches='tight')
                messagebox.showinfo("导出成功", f"图表已保存到: {filename}")


# Factory function
def get_gui_visualizer() -> Optional[WorkflowGUIVisualizer]:
    """Get GUI visualizer instance if available."""
    if GUI_AVAILABLE:
        return WorkflowGUIVisualizer()
    else:
        logger.warning("GUI libraries not available")
        return None


# Convenience function
def show_workflow_popup(task_description: str, workflow_type: str = "standard", 
                       workflow_data: Dict = None) -> bool:
    """
    Show workflow in popup window.
    
    Args:
        task_description: Task description
        workflow_type: Workflow type
        workflow_data: Optional workflow data
        
    Returns:
        True if popup was shown, False if GUI not available
    """
    try:
        visualizer = get_gui_visualizer()
        if visualizer:
            visualizer.show_workflow_popup(task_description, workflow_type, workflow_data)
            return True
        else:
            logger.warning("GUI visualization not available")
            return False
    except Exception as e:
        logger.error(f"Failed to show popup: {e}")
        return False
