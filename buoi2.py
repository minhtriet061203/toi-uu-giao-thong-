import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import time
from threading import Thread

class InteractivePathfindingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mô phỏng Thuật toán Tìm đường Dijkstra vs A*")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Grid setup
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = []  # Start with no obstacles, user can add them
        
        # Algorithm results
        self.dijkstra_explored = []
        self.dijkstra_path = []
        self.astar_explored = []
        self.astar_path = []
        
        # UI state
        self.dijkstra_cells_explored = 0
        self.dijkstra_path_length = 0
        self.dijkstra_time = 0
        self.astar_cells_explored = 0
        self.astar_path_length = 0
        self.astar_time = 0
        
        # Cell references for interaction
        self.dijkstra_cell_refs = {}
        self.astar_cell_refs = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="🤖 Mô phỏng Thuật toán Tìm đường Dijkstra vs A*", 
                              font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#333')
        title_label.pack()
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(pady=10)
        
        self.complete_btn = tk.Button(control_frame, text="✓ HOÀN THÀNH", 
                                     bg='#28a745', fg='white', font=('Arial', 10, 'bold'),
                                     padx=15, pady=5, command=self.run_both_algorithms)
        self.complete_btn.pack(side=tk.LEFT, padx=5)
        
        self.dijkstra_btn = tk.Button(control_frame, text="⚫ CHẠY DIJKSTRA", 
                                     bg='#007bff', fg='white', font=('Arial', 10, 'bold'),
                                     padx=15, pady=5, command=self.run_dijkstra_animation)
        self.dijkstra_btn.pack(side=tk.LEFT, padx=5)
        
        self.astar_btn = tk.Button(control_frame, text="⭐ CHẠY A*", 
                                  bg='#28a745', fg='white', font=('Arial', 10, 'bold'),
                                  padx=15, pady=5, command=self.run_astar_animation)
        self.astar_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(control_frame, text="🔄 ĐẶT LẠI", 
                                  bg='#dc3545', fg='white', font=('Arial', 10, 'bold'),
                                  padx=15, pady=5, command=self.reset)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        # Dijkstra section
        dijkstra_frame = tk.Frame(main_frame, bg='#f0f0f0')
        dijkstra_frame.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)
        
        dijkstra_title = tk.Label(dijkstra_frame, text="Thuật toán Dijkstra", 
                                 font=('Arial', 16, 'bold'), bg='#8e44ad', fg='white', 
                                 pady=10)
        dijkstra_title.pack(fill=tk.X)
        
        self.dijkstra_canvas = tk.Canvas(dijkstra_frame, width=350, height=350, 
                                        bg='white', highlightthickness=2, 
                                        highlightbackground='#ddd')
        self.dijkstra_canvas.pack(pady=10)
        self.dijkstra_canvas.bind("<Button-1>", lambda e: self.on_canvas_click(e, "dijkstra"))
        
        # Dijkstra stats frame
        dijkstra_stats_frame = tk.Frame(dijkstra_frame, bg='#e8f4fd', relief=tk.SOLID, bd=1)
        dijkstra_stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        stats_title = tk.Label(dijkstra_stats_frame, text="📊 Thống kê Dijkstra:", 
                              font=('Arial', 12, 'bold'), bg='#e8f4fd')
        stats_title.pack(anchor=tk.W, padx=10, pady=5)
        
        self.dijkstra_explored_label = tk.Label(dijkstra_stats_frame, text="Ô đã khám phá: 0", 
                                               font=('Arial', 11), bg='#e8f4fd')
        self.dijkstra_explored_label.pack(anchor=tk.W, padx=20)
        
        self.dijkstra_path_label = tk.Label(dijkstra_stats_frame, text="Độ dài đường đi: 0", 
                                           font=('Arial', 11), bg='#e8f4fd')
        self.dijkstra_path_label.pack(anchor=tk.W, padx=20)
        
        self.dijkstra_time_label = tk.Label(dijkstra_stats_frame, text="Thời gian: 0ms", 
                                           font=('Arial', 11), bg='#e8f4fd')
        self.dijkstra_time_label.pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Dijkstra instruction
        self.dijkstra_instruction = tk.Label(dijkstra_frame, text='Nhấn "Chạy Dijkstra" để bắt đầu', 
                                            font=('Arial', 11), bg='#d4edda', fg='#155724',
                                            relief=tk.SOLID, bd=1, pady=5)
        self.dijkstra_instruction.pack(fill=tk.X, padx=5, pady=5)
        
        # A* section
        astar_frame = tk.Frame(main_frame, bg='#f0f0f0')
        astar_frame.pack(side=tk.RIGHT, padx=20, fill=tk.BOTH, expand=True)
        
        astar_title = tk.Label(astar_frame, text="Thuật toán A*", 
                              font=('Arial', 16, 'bold'), bg='#e67e22', fg='white', 
                              pady=10)
        astar_title.pack(fill=tk.X)
        
        self.astar_canvas = tk.Canvas(astar_frame, width=350, height=350, 
                                     bg='white', highlightthickness=2, 
                                     highlightbackground='#ddd')
        self.astar_canvas.pack(pady=10)
        self.astar_canvas.bind("<Button-1>", lambda e: self.on_canvas_click(e, "astar"))
        
        # A* stats frame
        astar_stats_frame = tk.Frame(astar_frame, bg='#e8f4fd', relief=tk.SOLID, bd=1)
        astar_stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        astar_stats_title = tk.Label(astar_stats_frame, text="📊 Thống kê A*:", 
                                    font=('Arial', 12, 'bold'), bg='#e8f4fd')
        astar_stats_title.pack(anchor=tk.W, padx=10, pady=5)
        
        self.astar_explored_label = tk.Label(astar_stats_frame, text="Ô đã khám phá: 0", 
                                            font=('Arial', 11), bg='#e8f4fd')
        self.astar_explored_label.pack(anchor=tk.W, padx=20)
        
        self.astar_path_label = tk.Label(astar_stats_frame, text="Độ dài đường đi: 0", 
                                        font=('Arial', 11), bg='#e8f4fd')
        self.astar_path_label.pack(anchor=tk.W, padx=20)
        
        self.astar_time_label = tk.Label(astar_stats_frame, text="Thời gian: 0ms", 
                                        font=('Arial', 11), bg='#e8f4fd')
        self.astar_time_label.pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # A* instruction
        self.astar_instruction = tk.Label(astar_frame, text='Nhấn "Chạy A*" để bắt đầu', 
                                         font=('Arial', 11), bg='#d4edda', fg='#155724',
                                         relief=tk.SOLID, bd=1, pady=5)
        self.astar_instruction.pack(fill=tk.X, padx=5, pady=5)
        
        # Legend frame
        legend_frame = tk.Frame(self.root, bg='#f0f0f0')
        legend_frame.pack(pady=10)
        
        legend_items = [
            ("Điểm xuất phát", "#28a745"),
            ("Điểm đích", "#dc3545"),
            ("Vật cản", "#343a40"),
            ("Đã khám phá", "#b8daff"),
            ("Đường đi tối ưu", "#ffc107"),
            ("Đang xử lý", "#fd7e14")
        ]
        
        for i, (label, color) in enumerate(legend_items):
            item_frame = tk.Frame(legend_frame, bg='#f0f0f0')
            item_frame.pack(side=tk.LEFT, padx=10)
            
            color_box = tk.Label(item_frame, text="  ", bg=color, relief=tk.SOLID, bd=1)
            color_box.pack(side=tk.LEFT)
            
            text_label = tk.Label(item_frame, text=label, font=('Arial', 10), bg='#f0f0f0')
            text_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Instructions
        instruction_frame = tk.Frame(self.root, bg='#f0f0f0')
        instruction_frame.pack(pady=(10, 20))
        
        instruction_text = "💡 Hướng dẫn: Click vào các ô trắng để đặt/xóa vật cản. Điểm S (0,0) và G (4,4) không thể thay đổi."
        instruction_label = tk.Label(instruction_frame, text=instruction_text, 
                                    font=('Arial', 11), bg='#fff3cd', fg='#856404',
                                    relief=tk.SOLID, bd=1, padx=10, pady=5)
        instruction_label.pack()
        
        # Initialize grids
        self.draw_initial_grid(self.dijkstra_canvas, "dijkstra")
        self.draw_initial_grid(self.astar_canvas, "astar")
        
    def on_canvas_click(self, event, canvas_type):
        """Handle canvas click to add/remove obstacles"""
        canvas = self.dijkstra_canvas if canvas_type == "dijkstra" else self.astar_canvas
        
        # Calculate which cell was clicked
        cell_size = 60
        margin = 25
        
        x = event.x - margin
        y = event.y - margin
        
        if x < 0 or y < 0:
            return
            
        col = x // cell_size
        row = y // cell_size
        
        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            pos = (col, row)
            
            # Can't modify start or goal
            if pos == self.start or pos == self.goal:
                return
            
            # Toggle obstacle
            if pos in self.obstacles:
                self.obstacles.remove(pos)
            else:
                self.obstacles.append(pos)
            
            # Redraw both grids
            self.reset_results()
            self.draw_initial_grid(self.dijkstra_canvas, "dijkstra")
            self.draw_initial_grid(self.astar_canvas, "astar")
    
    def draw_initial_grid(self, canvas, canvas_type):
        canvas.delete("all")
        cell_size = 60
        margin = 25
        
        # Store cell references for interaction
        cell_refs = {}
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * cell_size + margin
                y1 = i * cell_size + margin
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Default cell
                color = 'white'
                text = ''
                text_color = 'black'
                border_color = '#ddd'
                border_width = 1
                
                # Start cell
                if (j, i) == self.start:
                    color = '#28a745'  # Green
                    text = 'S'
                    text_color = 'white'
                    border_color = '#1e7e34'
                    border_width = 2
                
                # Goal cell
                elif (j, i) == self.goal:
                    color = '#dc3545'  # Red
                    text = 'G'
                    text_color = 'white'
                    border_color = '#c82333'
                    border_width = 2
                
                # Obstacle cells
                elif (j, i) in self.obstacles:
                    color = '#343a40'  # Dark gray
                    text = '⚡'
                    text_color = '#ffc107'  # Yellow
                    border_color = '#23272b'
                    border_width = 2
                
                rect_id = canvas.create_rectangle(x1, y1, x2, y2, fill=color, 
                                                outline=border_color, width=border_width)
                if text:
                    text_id = canvas.create_text(x1 + cell_size//2, y1 + cell_size//2, 
                                               text=text, font=('Arial', 14, 'bold'), 
                                               fill=text_color)
                
                cell_refs[(j, i)] = (rect_id, text_color)
        
        if canvas_type == "dijkstra":
            self.dijkstra_cell_refs = cell_refs
        else:
            self.astar_cell_refs = cell_refs
    
    def draw_algorithm_result(self, canvas, canvas_type, explored, path):
        canvas.delete("all")
        cell_size = 60
        margin = 25
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * cell_size + margin
                y1 = i * cell_size + margin
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Default color
                color = 'white'
                text = ''
                text_color = 'black'
                border_color = '#ddd'
                border_width = 1
                
                # Check if cell is in path (highest priority)
                if (j, i) in path and (j, i) not in [self.start, self.goal]:
                    color = '#ffc107'  # Yellow for path
                    text_color = 'black'
                    border_color = '#e0a800'
                    border_width = 2
                
                # Check if cell is explored
                elif (j, i) in explored and (j, i) not in [self.start, self.goal]:
                    color = '#b8daff'  # Light blue for explored
                    text_color = 'black'
                    border_color = '#80bdff'
                    border_width = 1
                
                # Start cell (always visible)
                if (j, i) == self.start:
                    color = '#28a745'  # Green
                    text = 'S'
                    text_color = 'white'
                    border_color = '#1e7e34'
                    border_width = 2
                
                # Goal cell (always visible)
                elif (j, i) == self.goal:
                    color = '#dc3545'  # Red
                    text = 'G'
                    text_color = 'white'
                    border_color = '#c82333'
                    border_width = 2
                
                # Obstacle cells (always visible)
                elif (j, i) in self.obstacles:
                    color = '#343a40'  # Dark gray
                    text = '⚡'
                    text_color = '#ffc107'  # Yellow
                    border_color = '#23272b'
                    border_width = 2
                
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, 
                                      outline=border_color, width=border_width)
                if text:
                    canvas.create_text(x1 + cell_size//2, y1 + cell_size//2, 
                                     text=text, font=('Arial', 14, 'bold'), 
                                     fill=text_color)
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                (nx, ny) not in self.obstacles):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def dijkstra(self):
        start_time = time.time()
        
        distances = {(x, y): float('inf') for x in range(self.grid_size) for y in range(self.grid_size)}
        distances[self.start] = 0
        
        pq = [(0, self.start)]
        visited = set()
        parent = {}
        explored_order = []
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            explored_order.append(current)
            
            if current == self.goal:
                break
            
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                    
                new_dist = current_dist + 1
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct path
        path = []
        if self.goal in parent or self.goal == self.start:
            current = self.goal
            while current is not None:
                path.append(current)
                current = parent.get(current)
            path.reverse()
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return explored_order, path, execution_time
    
    def astar(self):
        start_time = time.time()
        
        g_score = {(x, y): float('inf') for x in range(self.grid_size) for y in range(self.grid_size)}
        g_score[self.start] = 0
        
        f_score = {(x, y): float('inf') for x in range(self.grid_size) for y in range(self.grid_size)}
        f_score[self.start] = self.manhattan_distance(self.start, self.goal)
        
        pq = [(f_score[self.start], self.start)]
        visited = set()
        parent = {}
        explored_order = []
        
        while pq:
            current_f, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            explored_order.append(current)
            
            if current == self.goal:
                break
            
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                    
                tentative_g = g_score[current] + 1
                
                if tentative_g < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, self.goal)
                    heapq.heappush(pq, (f_score[neighbor], neighbor))
        
        # Reconstruct path
        path = []
        if self.goal in parent or self.goal == self.start:
            current = self.goal
            while current is not None:
                path.append(current)
                current = parent.get(current)
            path.reverse()
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return explored_order, path, execution_time
    
    def run_dijkstra_animation(self):
        if not self.can_find_path():
            messagebox.showwarning("Cảnh báo", "Không có đường đi từ điểm xuất phát đến đích với vật cản hiện tại!")
            return
        
        self.dijkstra_explored, self.dijkstra_path, self.dijkstra_time = self.dijkstra()
        self.draw_algorithm_result(self.dijkstra_canvas, "dijkstra", 
                                 self.dijkstra_explored, self.dijkstra_path)
        
        # Update stats
        self.dijkstra_cells_explored = len(self.dijkstra_explored)
        self.dijkstra_path_length = len(self.dijkstra_path) - 1 if len(self.dijkstra_path) > 1 else 0
        
        self.dijkstra_explored_label.config(text=f"Ô đã khám phá: {self.dijkstra_cells_explored}")
        self.dijkstra_path_label.config(text=f"Độ dài đường đi: {self.dijkstra_path_length}")
        self.dijkstra_time_label.config(text=f"Thời gian: {self.dijkstra_time:.1f}ms")
        
        self.dijkstra_instruction.config(text="✅ Dijkstra đã hoàn thành!", bg='#d1ecf1', fg='#0c5460')
    
    def run_astar_animation(self):
        if not self.can_find_path():
            messagebox.showwarning("Cảnh báo", "Không có đường đi từ điểm xuất phát đến đích với vật cản hiện tại!")
            return
        
        self.astar_explored, self.astar_path, self.astar_time = self.astar()
        self.draw_algorithm_result(self.astar_canvas, "astar", 
                                 self.astar_explored, self.astar_path)
        
        # Update stats
        self.astar_cells_explored = len(self.astar_explored)
        self.astar_path_length = len(self.astar_path) - 1 if len(self.astar_path) > 1 else 0
        
        self.astar_explored_label.config(text=f"Ô đã khám phá: {self.astar_cells_explored}")
        self.astar_path_label.config(text=f"Độ dài đường đi: {self.astar_path_length}")
        self.astar_time_label.config(text=f"Thời gian: {self.astar_time:.1f}ms")
        
        self.astar_instruction.config(text="✅ A* đã hoàn thành!", bg='#d1ecf1', fg='#0c5460')
    
    def run_both_algorithms(self):
        if not self.can_find_path():
            messagebox.showwarning("Cảnh báo", "Không có đường đi từ điểm xuất phát đến đích với vật cản hiện tại!")
            return
        
        self.run_dijkstra_animation()
        self.run_astar_animation()
        
        # Show detailed comparison
        if self.dijkstra_explored and self.astar_explored:
            self.show_comparison_window()
    
    def can_find_path(self):
        """Check if there's a path from start to goal using simple BFS"""
        from collections import deque
        
        queue = deque([self.start])
        visited = {self.start}
        
        while queue:
            current = queue.popleft()
            
            if current == self.goal:
                return True
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def show_comparison_window(self):
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("📊 So sánh Chi tiết")
        comparison_window.geometry("500x400")
        comparison_window.configure(bg='#f0f0f0')
        comparison_window.resizable(False, False)
        
        # Title
        title_label = tk.Label(comparison_window, text="📊 Kết quả So sánh Chi tiết", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        # Comparison table frame
        table_frame = tk.Frame(comparison_window, bg='white', relief=tk.SOLID, bd=1)
        table_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Headers
        headers = ["Tiêu chí", "Dijkstra", "A*", "Chênh lệch"]
        for i, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, font=('Arial', 12, 'bold'), 
                           bg='#e9ecef', relief=tk.SOLID, bd=1, padx=10, pady=5)
            label.grid(row=0, column=i, sticky='ew')
        
        # Data rows
        data = [
            ("Ô đã khám phá", self.dijkstra_cells_explored, self.astar_cells_explored, 
             self.dijkstra_cells_explored - self.astar_cells_explored),
            ("Độ dài đường đi", self.dijkstra_path_length, self.astar_path_length, 
             self.dijkstra_path_length - self.astar_path_length),
            ("Thời gian (ms)", f"{self.dijkstra_time:.1f}", f"{self.astar_time:.1f}", 
             f"{self.dijkstra_time - self.astar_time:.1f}")
        ]
        
        for row, (criterion, dij_val, ast_val, diff) in enumerate(data, 1):
            tk.Label(table_frame, text=criterion, font=('Arial', 11), 
                    bg='white', relief=tk.SOLID, bd=1, padx=10, pady=5).grid(row=row, column=0, sticky='ew')
            tk.Label(table_frame, text=str(dij_val), font=('Arial', 11), 
                    bg='#fff3cd', relief=tk.SOLID, bd=1, padx=10, pady=5).grid(row=row, column=1, sticky='ew')
            tk.Label(table_frame, text=str(ast_val), font=('Arial', 11), 
                    bg='#d1ecf1', relief=tk.SOLID, bd=1, padx=10, pady=5).grid(row=row, column=2, sticky='ew')
            
            # Color diff based on value
            diff_color = '#d4edda' if str(diff).startswith('-') or diff == 0 else '#f8d7da'
            tk.Label(table_frame, text=str(diff), font=('Arial', 11), 
                    bg=diff_color, relief=tk.SOLID, bd=1, padx=10, pady=5).grid(row=row, column=3, sticky='ew')
        
        # Configure column weights
        for i in range(4):
            table_frame.columnconfigure(i, weight=1)
        
        # Analysis
        analysis_frame = tk.Frame(comparison_window, bg='#f0f0f0')
        analysis_frame.pack(padx=20, pady=10, fill=tk.X)
        
        analysis_title = tk.Label(analysis_frame, text="🔍 Phân tích:", 
                                 font=('Arial', 12, 'bold'), bg='#f0f0f0')
        analysis_title.pack(anchor=tk.W)
        
        if self.astar_cells_explored < self.dijkstra_cells_explored:
            analysis_text = f"• A* hiệu quả hơn, khám phá ít hơn {self.dijkstra_cells_explored - self.astar_cells_explored} ô\n"
        else:
            analysis_text = "• Cả hai thuật toán khám phá số ô tương đương\n"
        
        if self.dijkstra_path_length == self.astar_path_length:
            analysis_text += "• Cả hai đều tìm được đường đi tối ưu\n"
        
        analysis_text += "• A* sử dụng heuristic Manhattan để hướng tìm kiếm"
        
        analysis_label = tk.Label(analysis_frame, text=analysis_text, 
                                 font=('Arial', 10), bg='#f0f0f0', justify=tk.LEFT)
        analysis_label.pack(anchor=tk.W, padx=10)
        
        # Close button
        close_btn = tk.Button(comparison_window, text="Đóng", 
                             bg='#6c757d', fg='white', font=('Arial', 11, 'bold'),
                             padx=20, pady=5, command=comparison_window.destroy)
        close_btn.pack(pady=10)
    
    def reset_results(self):
        """Reset algorithm results"""
        self.dijkstra_explored = []
        self.dijkstra_path = []
        self.astar_explored = []
        self.astar_path = []
        
        # Reset stats
        self.dijkstra_cells_explored = 0
        self.dijkstra_path_length = 0
        self.dijkstra_time = 0
        self.astar_cells_explored = 0
        self.astar_path_length = 0
        self.astar_time = 0
        
        # Update labels
        self.dijkstra_explored_label.config(text="Ô đã khám phá: 0")
        self.dijkstra_path_label.config(text="Độ dài đường đi: 0")
        self.dijkstra_time_label.config(text="Thời gian: 0ms")
        
        self.astar_explored_label.config(text="Ô đã khám phá: 0")
        self.astar_path_label.config(text="Độ dài đường đi: 0")
        self.astar_time_label.config(text="Thời gian: 0ms")
        
        # Reset instructions
        self.dijkstra_instruction.config(text='Nhấn "Chạy Dijkstra" để bắt đầu', 
                                        bg='#d4edda', fg='#155724')
        self.astar_instruction.config(text='Nhấn "Chạy A*" để bắt đầu', 
                                     bg='#d4edda', fg='#155724')
    
    def reset(self):
        """Reset everything to initial state"""
        self.obstacles = []
        self.reset_results()
        
        self.draw_initial_grid(self.dijkstra_canvas, "dijkstra")
        self.draw_initial_grid(self.astar_canvas, "astar")
    
    def run(self):
        # Show initial instructions
        messagebox.showinfo("Hướng dẫn", 
                           "🎯 Hướng dẫn sử dụng:\n\n" +
                           "1. Click vào các ô trắng để đặt/xóa vật cản\n" +
                           "2. Nhấn 'Chạy Dijkstra' hoặc 'Chạy A*' để xem thuật toán\n" +
                           "3. Nhấn 'Hoàn thành' để chạy cả hai và so sánh\n" +
                           "4. Nhấn 'Đặt lại' để xóa tất cả vật cản\n\n" +
                           "Điểm S (0,0) và G (4,4) không thể thay đổi.")
        
        self.root.mainloop()

if __name__ == "__main__":
    app = InteractivePathfindingGUI()
    app.run()