import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygame
import sys
import time
import math


class TrafficNetwork:
    def __init__(self, congestion_data):
        """
        Khoi tao mang luoi giao thong voi 6 nut giao theo hinh da giac
        congestion_data: dictionary chua chi so tac nghen cho moi duong tai moi nut giao
        """
        self.congestion_data = congestion_data
        # Update to 6 intersections as per diagram (A, B, C, D, E, F)
        self.intersections = ["Nut 1", "Nut 2", "Nut 3", "Nut 4", "Nut 5", "Nut 6"]
        self.roads_per_intersection = ["A", "B", "C", "D"]

        # Thong so co ban
        self.min_green_time = 15  # Thoi gian den xanh toi thieu (giay)
        self.max_green_time = 90  # Thoi gian den xanh toi da (giay)
        self.amber_time = 3  # Thoi gian den vang (giay)
        self.clearance_time = 2  # Thoi gian giai phong giao lo (giay)

        # Ma tran khoang cach giua cac nut giao (don vi: met)
        # Based on polygon shape in diagram
        self.distance_matrix = {
            "Nut 1": {"Nut 1": 0, "Nut 2": 300, "Nut 3": 0, "Nut 4": 350, "Nut 5": 0, "Nut 6": 250},
            "Nut 2": {"Nut 1": 300, "Nut 2": 0, "Nut 3": 300, "Nut 4": 0, "Nut 5": 400, "Nut 6": 0},
            "Nut 3": {"Nut 1": 0, "Nut 2": 300, "Nut 3": 0, "Nut 4": 300, "Nut 5": 0, "Nut 6": 0},
            "Nut 4": {"Nut 1": 350, "Nut 2": 0, "Nut 3": 300, "Nut 4": 0, "Nut 5": 0, "Nut 6": 0},
            "Nut 5": {"Nut 1": 0, "Nut 2": 400, "Nut 3": 0, "Nut 4": 0, "Nut 5": 0, "Nut 6": 350},
            "Nut 6": {"Nut 1": 250, "Nut 2": 0, "Nut 3": 0, "Nut 4": 0, "Nut 5": 350, "Nut 6": 0}
        }

        # Ma tran ket noi giua cac nut giao
        # Value: (duong ra tu nut 1, duong vao nut 2)
        # Based on polygon shape in diagram
        self.connections = {
            "Nut 1": {"Nut 1": None, "Nut 2": ("B", "A"), "Nut 3": None, "Nut 4": ("C", "A"), "Nut 5": None, "Nut 6": ("D", "A")},
            "Nut 2": {"Nut 1": ("A", "B"), "Nut 2": None, "Nut 3": ("C", "A"), "Nut 4": None, "Nut 5": ("D", "A"), "Nut 6": None},
            "Nut 3": {"Nut 1": None, "Nut 2": ("A", "C"), "Nut 3": None, "Nut 4": ("C", "A"), "Nut 5": None, "Nut 6": None},
            "Nut 4": {"Nut 1": ("A", "C"), "Nut 2": None, "Nut 3": ("A", "C"), "Nut 4": None, "Nut 5": None, "Nut 6": None},
            "Nut 5": {"Nut 1": None, "Nut 2": ("A", "D"), "Nut 3": None, "Nut 4": None, "Nut 5": None, "Nut 6": ("C", "B")},
            "Nut 6": {"Nut 1": ("A", "D"), "Nut 2": None, "Nut 3": None, "Nut 4": None, "Nut 5": ("B", "C"), "Nut 6": None}
        }

        # Thiet lap cac nhom pha cho moi nut giao
        # (duong A,C di chung mot pha va B,D di chung pha khac)
        self.phase_groups = {
            intersection: [
                [f"{intersection}-A", f"{intersection}-C"],
                [f"{intersection}-B", f"{intersection}-D"]
            ] for intersection in self.intersections
        }

        # Van toc trung binh cua xe (m/s) - khoang 30km/h
        self.avg_speed = 8.33

        # Bien luu ket qua
        self.signal_timings = {}
        self.cycle_times = {}
        self.offset_times = {}

    def optimize_independent(self):
        """
        Toi uu hoa doc lap cho tung nut giao thong ma khong tinh den moi lien he
        """
        for intersection in self.intersections:
            # Lay du lieu tac nghen cho nut giao thong nay
            congestion_at_intersection = {}
            for road in self.roads_per_intersection:
                key = f"{intersection}-{road}"
                congestion_at_intersection[key] = self.congestion_data[key]

            # Toi uu hoa nut giao thong nay
            timings, cycle_time = self.optimize_single_intersection(congestion_at_intersection, intersection)

            # Luu ket qua
            self.signal_timings[intersection] = timings
            self.cycle_times[intersection] = cycle_time

        return self.signal_timings, self.cycle_times

    def optimize_network(self):
        """
        Toi uu hoa ca mang luoi giao thong voi 6 nut giao lien ket
        """
        # Buoc 1: Toi uu hoa doc lap cho tung nut giao
        self.optimize_independent()

        # Buoc 2: Tinh toan offset de tao ra lan song xanh
        self.calculate_offsets()

        # Buoc 3: Dieu chinh chu ky giua cac nut giao
        self.normalize_cycle_times()

        # Buoc 4: Dieu chinh lai thoi gian xanh sau khi co chu ky chung
        self.adjust_green_times()

        return self.signal_timings, self.cycle_times, self.offset_times

    def optimize_single_intersection(self, congestion_levels, intersection):
        """
        Thuat toan toi uu hoa thoi gian den giao thong cho 1 nut giao
        su dung phuong phap Webster ket hop voi Logic mo
        """
        # Lay nhom pha cho nut giao nay
        phase_groups = self.phase_groups[intersection]

        # 1. Tinh toan mat do trung binh cho moi pha
        phase_congestion = []
        for phase in phase_groups:
            # Lay gia tri tac nghen lon nhat trong pha (yeu to quyet dinh)
            max_congestion = max([congestion_levels[road] for road in phase])
            phase_congestion.append(max_congestion)

        # 2. Ap dung Logic mo de dieu chinh thoi gian den
        # Thiet lap cac bien ngon ngu cho Logic mo
        congestion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'congestion')
        green_time = ctrl.Consequent(np.arange(self.min_green_time, self.max_green_time + 1, 1), 'green_time')

        # Dinh nghia cac tap mo cho mat do tac nghen
        congestion['low'] = fuzz.trimf(congestion.universe, [0, 0, 0.3])
        congestion['medium'] = fuzz.trimf(congestion.universe, [0.2, 0.5, 0.8])
        congestion['high'] = fuzz.trimf(congestion.universe, [0.7, 1, 1])

        # Dinh nghia cac tap mo cho thoi gian den xanh
        green_time['short'] = fuzz.trimf(green_time.universe, [self.min_green_time, self.min_green_time, 35])
        green_time['medium'] = fuzz.trimf(green_time.universe, [30, 45, 60])
        green_time['long'] = fuzz.trimf(green_time.universe, [55, self.max_green_time, self.max_green_time])

        # Dinh nghia cac luat mo
        rule1 = ctrl.Rule(congestion['low'], green_time['short'])
        rule2 = ctrl.Rule(congestion['medium'], green_time['medium'])
        rule3 = ctrl.Rule(congestion['high'], green_time['long'])

        # Thiet lap he thong dieu khien mo
        green_time_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        green_time_simulator = ctrl.ControlSystemSimulation(green_time_ctrl)

        # 3. Ap dung cong thuc Webster de tinh toan chu ky toi uu
        lost_time_per_phase = self.amber_time + self.clearance_time
        total_lost_time = lost_time_per_phase * len(phase_groups)

        # Tinh toan ty le dong tac nghen cho moi pha
        y_values = [min(0.95 * congestion, 0.95) for congestion in phase_congestion]
        Y = sum(y_values)

        # Cong thuc Webster cho chu ky toi uu
        optimal_cycle = (1.5 * total_lost_time + 5) / (1 - Y)
        optimal_cycle = min(max(optimal_cycle, 60), 150)  # Gioi han chu ky tu 60-150s

        # 4. Phan bo thoi gian xanh cho moi pha
        effective_green_times = {}
        timings = {}

        for i, phase in enumerate(phase_groups):
            # Su dung Logic mo de tinh thoi gian xanh ban dau
            green_time_simulator.input['congestion'] = phase_congestion[i]
            green_time_simulator.compute()
            initial_green = green_time_simulator.output['green_time']

            # Dieu chinh theo cong thuc Webster
            effective_green = (y_values[i] / Y) * (optimal_cycle - total_lost_time)

            # Lay gia tri trung binh co trong so
            final_green = 0.7 * effective_green + 0.3 * initial_green

            # Dam bao thoi gian xanh nam trong gioi han
            final_green = min(max(final_green, self.min_green_time), self.max_green_time)
            effective_green_times[i] = round(final_green)

        # 5. Tinh toan thoi gian do cho moi pha
        for i, phase in enumerate(phase_groups):
            for road in phase:
                other_phases_green = sum([effective_green_times[j] for j in range(len(phase_groups)) if j != i])
                red_time = other_phases_green + lost_time_per_phase * (len(phase_groups) - 1)

                timings[road] = {
                    "green_time": effective_green_times[i],
                    "red_time": round(red_time),
                    "amber_time": self.amber_time,
                    "clearance_time": self.clearance_time
                }

        # 6. Tinh toan va tra ve thong tin chu ky
        cycle_time = sum(
            [effective_green_times[i] + self.amber_time for i in range(len(phase_groups))]) + total_lost_time

        return timings, round(cycle_time)

    def calculate_offsets(self):
        """
        Tinh toan do lech pha (offset) giua cac nut giao thong de tao ra lan song xanh
        """
        # Chon nut giao thong co luu luong cao nhat lam nut tham chieu (offset = 0)
        total_congestion_per_intersection = {}
        for intersection in self.intersections:
            total = sum(self.congestion_data[f"{intersection}-{road}"] for road in self.roads_per_intersection)
            total_congestion_per_intersection[intersection] = total

        reference_intersection = max(total_congestion_per_intersection, key=total_congestion_per_intersection.get)

        # Dat offset cua nut tham chieu bang 0
        self.offset_times = {reference_intersection: 0}

        # Tinh offset cho cac nut khac
        for intersection in self.intersections:
            if intersection == reference_intersection:
                continue

            # Tim duong lien ket giua nut tham chieu va nut nay
            connection = self.connections[reference_intersection][intersection]
            if connection is None:
                # Neu khong co lien ket truc tiep, tinh duong di gan nhat
                path = self.find_shortest_path(reference_intersection, intersection)
                if not path:
                    # Neu khong co duong di, dat offset bang 0
                    self.offset_times[intersection] = 0
                    continue

                # Tinh tong thoi gian di chuyen
                total_travel_time = 0
                for i in range(len(path) - 1):
                    distance = self.distance_matrix[path[i]][path[i + 1]]
                    total_travel_time += distance / self.avg_speed

                # Tinh offset dua tren thoi gian di chuyen
                # Muc tieu: Xe den nut tiep theo dung khi den xanh bat dau
                cycle_time = self.cycle_times[intersection]
                offset = total_travel_time % cycle_time
                self.offset_times[intersection] = round(offset)
            else:
                # Neu co lien ket truc tiep, tinh offset dua tren thoi gian di chuyen
                out_road, in_road = connection
                distance = self.distance_matrix[reference_intersection][intersection]
                travel_time = distance / self.avg_speed

                # Tinh thoi diem bat dau den xanh cho duong vao cua nut dich
                in_road_key = f"{intersection}-{in_road}"
                phase_with_in_road = None
                for phase in self.phase_groups[intersection]:
                    if in_road_key in phase:
                        phase_with_in_road = phase
                        break

                # Thoi diem bat dau den xanh cho in_road o nut dich
                if phase_with_in_road == self.phase_groups[intersection][0]:
                    # Neu la pha 1
                    green_start_time_dest = 0
                else:
                    # Neu la pha 2
                    green_start_time_dest = self.signal_timings[intersection][phase_with_in_road[0]][
                                                "green_time"] + self.amber_time

                # Thoi diem bat dau den xanh cho out_road o nut nguon
                out_road_key = f"{reference_intersection}-{out_road}"
                phase_with_out_road = None
                for phase in self.phase_groups[reference_intersection]:
                    if out_road_key in phase:
                        phase_with_out_road = phase
                        break

                if phase_with_out_road == self.phase_groups[reference_intersection][0]:
                    # Neu la pha 1
                    green_start_time_source = 0
                else:
                    # Neu la pha 2
                    green_start_time_source = self.signal_timings[reference_intersection][phase_with_out_road[0]][
                                                  "green_time"] + self.amber_time

                # Tinh offset: thoi diem den xanh cua nut dich = thoi diem den xanh cua nut nguon + thoi gian di chuyen
                cycle_time = self.cycle_times[intersection]
                ideal_offset = (green_start_time_source + travel_time - green_start_time_dest) % cycle_time
                self.offset_times[intersection] = round(ideal_offset)

        return self.offset_times

    def normalize_cycle_times(self):
        """
        Dieu chinh chu ky cua cac nut giao thong de chung dong bo voi nhau
        """
        # Tim chu ky lon nhat trong cac nut giao
        max_cycle = max(self.cycle_times.values())

        # Dat chu ky chung bang chu ky lon nhat
        for intersection in self.intersections:
            self.cycle_times[intersection] = max_cycle

        return self.cycle_times

    def adjust_green_times(self):
        """
        Dieu chinh lai thoi gian den xanh sau khi co chu ky chung
        """
        for intersection in self.intersections:
            cycle_time = self.cycle_times[intersection]
            phase_groups = self.phase_groups[intersection]

            # Tinh tong thoi gian xanh hien tai
            current_green_times = [self.signal_timings[intersection][phase[0]]["green_time"] for phase in phase_groups]
            total_current_green = sum(current_green_times)

            # Tinh thoi gian mat di trong chu ky
            lost_time_per_phase = self.amber_time + self.clearance_time
            total_lost_time = lost_time_per_phase * len(phase_groups)

            # Tinh tong thoi gian xanh co the co trong chu ky moi
            available_green_time = cycle_time - total_lost_time

            # Tinh ty le dieu chinh
            adjustment_ratio = available_green_time / total_current_green

            # Dieu chinh thoi gian xanh cho moi pha
            for i, phase in enumerate(phase_groups):
                new_green_time = round(current_green_times[i] * adjustment_ratio)
                new_green_time = max(new_green_time, self.min_green_time)  # Dam bao thoi gian xanh toi thieu

                # Cap nhat thoi gian xanh moi
                for road in phase:
                    self.signal_timings[intersection][road]["green_time"] = new_green_time

            # Tinh lai thoi gian do cho moi pha
            for i, phase in enumerate(phase_groups):
                for road in phase:
                    # Thoi gian xanh cua cac pha khac + thoi gian vang cua cac pha
                    other_phases_green = sum([self.signal_timings[intersection][phase_groups[j][0]]["green_time"]
                                              for j in range(len(phase_groups)) if j != i])
                    other_phases_amber = (len(phase_groups) - 1) * self.amber_time

                    # Thoi gian do = chu ky - thoi gian xanh cua pha nay - thoi gian vang cua pha nay
                    red_time = cycle_time - self.signal_timings[intersection][road]["green_time"] - self.amber_time
                    self.signal_timings[intersection][road]["red_time"] = red_time

        return self.signal_timings

    def find_shortest_path(self, start, end):
        """
        Tim duong di ngan nhat giua hai nut giao thong
        """
        # Thuat toan Dijkstra
        distances = {intersection: float('infinity') for intersection in self.intersections}
        distances[start] = 0

        unvisited = list(self.intersections)
        previous = {intersection: None for intersection in self.intersections}

        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])

            if current == end:
                # Tao duong di tu diem cuoi ve diem dau
                path = []
                while current:
                    path.append(current)
                    current = previous[current]
                # Dao nguoc duong di
                return path[::-1]

            if distances[current] == float('infinity'):
                # Khong co duong di den diem cuoi
                break

            # Loai bo nut hien tai khoi danh sach chua xet
            unvisited.remove(current)

            # Kiem tra cac nut ke
            for neighbor in self.intersections:
                if neighbor == current:
                    continue

                if self.connections[current][neighbor]:
                    # Co duong di tu current den neighbor
                    distance = self.distance_matrix[current][neighbor]
                    alternative_route = distances[current] + distance

                    if alternative_route < distances[neighbor]:
                        distances[neighbor] = alternative_route
                        previous[neighbor] = current

        # Neu khong tim thay duong di
        return []


def run_network_simulation(network):
    """
    Mo phong truc quan hoat dong cua mang luoi den giao thong
    """
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Mo phong Mang luoi Den Giao thong Thong minh - Da giac")
    clock = pygame.time.Clock()

    # Font chu
    font_small = pygame.font.Font(None, 24)
    font_medium = pygame.font.Font(None, 32)
    font_large = pygame.font.Font(None, 40)

    # Mau sac
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (150, 150, 150)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (173, 216, 230)

    # Vi tri cac nut giao thong theo hinh da giac
    intersection_positions = {
        # Rectangle ABCD corners + points E and F
        "Nut 1": (300, 250),  # A - top left of rectangle
        "Nut 2": (600, 250),  # F - top 
        "Nut 3": (900, 250),  # D - top right of rectangle
        "Nut 4": (900, 550),  # C - bottom right of rectangle
        "Nut 5": (600, 650),  # E - bottom
        "Nut 6": (300, 550)   # B - bottom left of rectangle
    }

    # Kich thuoc cua moi nut giao
    intersection_size = 120

    # Vi tri cac den giao thong cho moi nut giao
    traffic_light_positions = {}
    for intersection, center in intersection_positions.items():
        x, y = center
        traffic_light_positions[f"{intersection}-A"] = (x - intersection_size // 2, y)
        traffic_light_positions[f"{intersection}-B"] = (x, y - intersection_size // 2)
        traffic_light_positions[f"{intersection}-C"] = (x + intersection_size // 2, y)
        traffic_light_positions[f"{intersection}-D"] = (x, y + intersection_size // 2)

    # Lay thong tin tu network
    signal_timings = network.signal_timings
    cycle_times = network.cycle_times
    offset_times = network.offset_times
    congestion_data = network.congestion_data

    # Ve nut giao thong
    def draw_intersection(center, size, intersection_name, time_in_cycle):
        x, y = center

        # Ve hinh vuong dai dien cho nut giao
        pygame.draw.rect(screen, BLACK, (x - size // 2, y - size // 2, size, size))

        # Ve duong doc va ngang qua nut giao
        road_width = 30
        pygame.draw.rect(screen, GRAY, (x - size // 2, y - road_width // 2, size, road_width))  # Duong ngang
        pygame.draw.rect(screen, GRAY, (x - road_width // 2, y - size // 2, road_width, size))  # Duong doc

        # Ve vach ke duong
        line_width = 2
        line_length = 10
        line_gap = 10

        # Vach ke duong ngang
        for i in range(size // 2 // (line_length + line_gap)):
            # Ben trai
            pygame.draw.rect(screen, WHITE,
                             (x - size // 2 + i * (line_length + line_gap),
                              y - line_width // 2,
                              line_length, line_width))
            # Ben phai
            pygame.draw.rect(screen, WHITE,
                             (x + i * (line_length + line_gap),
                              y - line_width // 2,
                              line_length, line_width))

        # Vach ke duong doc
        for i in range(size // 2 // (line_length + line_gap)):
            # Ben tren
            pygame.draw.rect(screen, WHITE,
                             (x - line_width // 2,
                              y - size // 2 + i * (line_length + line_gap),
                              line_width, line_length))
            # Ben duoi
            pygame.draw.rect(screen, WHITE,
                             (x - line_width // 2,
                              y + i * (line_length + line_gap),
                              line_width, line_length))

        # Ve ten nut giao
        text = font_medium.render(intersection_name, True, WHITE)
        screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

        # Ve thoi gian chu ky
        cycle_time = cycle_times[intersection_name]
        offset = offset_times[intersection_name]
        adjusted_time = (time_in_cycle - offset) % cycle_time

        text_cycle = font_small.render(f"Chu ky: {cycle_time}s", True, WHITE)
        text_current = font_small.render(f"Thoi gian: {adjusted_time:.1f}s", True, WHITE)
        text_offset = font_small.render(f"Offset: {offset}s", True, WHITE)

        screen.blit(text_cycle, (x - size // 2, y + size // 2 + 5))
        screen.blit(text_current, (x - size // 2, y + size // 2 + 25))
        screen.blit(text_offset, (x - size // 2, y + size // 2 + 45))

    # Ve den giao thong
    def draw_traffic_light(position, road_id, is_green):
        x, y = position
        intersection, road = road_id.split("-")

        # Tim vi tri cua den dua tren duong
        if road == "A":  # Den ben trai
            light_x = x - 20
            light_y = y
        elif road == "B":  # Den ben tren
            light_x = x
            light_y = y - 20
        elif road == "C":  # Den ben phai
            light_x = x + 20
            light_y = y
        elif road == "D":  # Den ben duoi
            light_x = x
            light_y = y + 20

        # Ve hinh tron dai dien cho den
        if is_green:
            pygame.draw.circle(screen, GREEN, (light_x, light_y), 8)
        else:
            pygame.draw.circle(screen, RED, (light_x, light_y), 8)

        # Ve ten cua duong
        text = font_small.render(road, True, WHITE)
        if road == "A":
            screen.blit(text, (light_x - 30, light_y - 10))
        elif road == "B":
            screen.blit(text, (light_x - 5, light_y - 30))
        elif road == "C":
            screen.blit(text, (light_x + 20, light_y - 10))
        elif road == "D":
            screen.blit(text, (light_x - 5, light_y + 15))

    # Ve duong lien ket giua cac nut giao
    def draw_connections():
        # Ve duong lien ket giua cac nut giao thong theo hinh da giac
        # First draw the external polygon connections
        polygon_points = [
            intersection_positions["Nut 1"],  # A
            intersection_positions["Nut 2"],  # F
            intersection_positions["Nut 3"],  # D
            intersection_positions["Nut 4"],  # C
            intersection_positions["Nut 5"],  # E
            intersection_positions["Nut 6"],  # B
        ]
        
        # Draw the polygon outline
        for i in range(len(polygon_points)):
            next_i = (i + 1) % len(polygon_points)
            
            # Only draw direct connections that exist in the network
            from_node = f"Nut {i+1}"
            to_node = f"Nut {next_i+1}"
            
            if network.connections[from_node][to_node] is not None or network.connections[to_node][from_node] is not None:
                pygame.draw.line(screen, LIGHT_BLUE, polygon_points[i], polygon_points[next_i], 4)
                
                # Calculate midpoint for connection label
                mid_x = (polygon_points[i][0] + polygon_points[next_i][0]) // 2
                mid_y = (polygon_points[i][1] + polygon_points[next_i][1]) // 2
                
                # Get connection info if it exists
                if network.connections[from_node][to_node] is not None:
                    out_road, in_road = network.connections[from_node][to_node]
                    text = font_small.render(f"{out_road} -> {in_road}", True, WHITE)
                    screen.blit(text, (mid_x, mid_y))
                elif network.connections[to_node][from_node] is not None:
                    out_road, in_road = network.connections[to_node][from_node]
                    text = font_small.render(f"{out_road} -> {in_road}", True, WHITE)
                    screen.blit(text, (mid_x, mid_y))
        
        # Draw the rectangle (connections between A-D-C-B)
        pygame.draw.line(screen, BLUE, intersection_positions["Nut 1"], intersection_positions["Nut 3"], 2)  # A to D
        pygame.draw.line(screen, BLUE, intersection_positions["Nut 3"], intersection_positions["Nut 4"], 2)  # D to C
        pygame.draw.line(screen, BLUE, intersection_positions["Nut 4"], intersection_positions["Nut 6"], 2)  # C to B
        pygame.draw.line(screen, BLUE, intersection_positions["Nut 6"], intersection_positions["Nut 1"], 2)  # B to A

    # Ve xe tren duong lien ket
    def draw_vehicles_on_connections(simulation_time):
        # Draw vehicles on each valid connection
        for i, intersection1 in enumerate(network.intersections):
            for j, intersection2 in enumerate(network.intersections):
                if i != j and network.connections[intersection1][intersection2]:
                    pos1 = intersection_positions[intersection1]
                    pos2 = intersection_positions[intersection2]

                    # Lay thong tin lien ket
                    out_road, in_road = network.connections[intersection1][intersection2]
                    out_road_key = f"{intersection1}-{out_road}"
                    in_road_key = f"{intersection2}-{in_road}"

                    # Kiem tra den out_road co mau xanh khong
                    out_road_signal_time = (simulation_time - offset_times[intersection1]) % cycle_times[intersection1]
                    is_out_road_green = False

                    # Xac dinh pha cua out_road
                    phase_with_out_road = None
                    for phase in network.phase_groups[intersection1]:
                        if out_road_key in phase:
                            phase_with_out_road = phase
                            break

                    # Tinh thoi gian den xanh cua out_road
                    if phase_with_out_road == network.phase_groups[intersection1][0]:
                        # Neu la pha 1
                        is_out_road_green = out_road_signal_time < signal_timings[intersection1][out_road_key][
                            "green_time"]
                    else:
                        # Neu la pha 2
                        green_start = signal_timings[intersection1][network.phase_groups[intersection1][0][0]][
                                          "green_time"] + network.amber_time
                        is_out_road_green = green_start <= out_road_signal_time < green_start + \
                                            signal_timings[intersection1][out_road_key]["green_time"]

                    # Chi ve xe khi den xanh
                    if is_out_road_green:
                        # Tinh huong di chuyen
                        direction = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                        length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                        normalized = (direction[0] / length, direction[1] / length)

                        # Tinh so luong xe dua tren mat do tac nghen
                        congestion = congestion_data[out_road_key]
                        num_vehicles = int(congestion * 15)

                        # Ve cac xe tren duong noi
                        for k in range(num_vehicles):
                            # Tinh vi tri xe dua tren thoi gian
                            progress = (simulation_time * speed_factor / 10 + k / num_vehicles) % 1
                            vehicle_x = pos1[0] + progress * direction[0]
                            vehicle_y = pos1[1] + progress * direction[1]

                            # Ve hinh chu nhat dai dien cho xe
                            vehicle_size = 8
                            pygame.draw.rect(screen, YELLOW,
                                             (vehicle_x - vehicle_size // 2,
                                              vehicle_y - vehicle_size // 2,
                                              vehicle_size, vehicle_size))

    # Ve bang thong tin so sanh
    def draw_comparison_table():
        x, y = 20, 20
        width, height = 400, 150

        # Ve nen bang
        pygame.draw.rect(screen, (50, 50, 50), (x, y, width, height), border_radius=5)

        # Tieu de
        text_title = font_medium.render("So sanh mo hinh toi uu", True, WHITE)
        screen.blit(text_title, (x + 10, y + 10))

        # Thong tin chu ky
        independent_cycles = [cycle_times[intersection] for intersection in network.intersections]
        avg_cycle = sum(independent_cycles) / len(independent_cycles)
        max_cycle = max(independent_cycles)

        y_offset = 40
        text1 = font_small.render(f"Chu ky chung: {max_cycle}s", True, WHITE)
        text2 = font_small.render(f"Chu ky trung binh (neu doc lap): {avg_cycle:.1f}s", True, WHITE)
        text3 = font_small.render(f"Cai thien bang lan song xanh va offset", True, WHITE)

        screen.blit(text1, (x + 10, y + y_offset))
        screen.blit(text2, (x + 10, y + y_offset + 25))
        screen.blit(text3, (x + 10, y + y_offset + 50))

        # Hieu qua
        text4 = font_small.render("Tong do lech offset: " +
                                  str(sum(offset_times.values())) + "s", True, WHITE)
        screen.blit(text4, (x + 10, y + y_offset + 75))

        text5 = font_small.render("Hieu qua: Giam thoi gian dung lai giua cac nut giao", True, WHITE)
        screen.blit(text5, (x + 10, y + y_offset + 100))

    # Ve huong dan su dung
    def draw_instructions():
        x, y = 750, 20
        width, height = 430, 120

        # Ve nen bang
        pygame.draw.rect(screen, (50, 50, 50), (x, y, width, height), border_radius=5)

        # Tieu de
        text_title = font_medium.render("Huong dan su dung", True, WHITE)
        screen.blit(text_title, (x + 10, y + 10))

        # Cac phim dieu khien
        text1 = font_small.render("SPACE: Tam dung/Tiep tuc", True, WHITE)
        text2 = font_small.render("UP/DOWN: Tang/Giam toc do mo phong", True, WHITE)
        text3 = font_small.render("R: Khoi dong lai mo phong", True, WHITE)
        text4 = font_small.render("ESC: Thoat", True, WHITE)

        screen.blit(text1, (x + 10, y + 40))
        screen.blit(text2, (x + 10, y + 60))
        screen.blit(text3, (x + 10, y + 80))
        screen.blit(text4, (x + 10, y + 100))

    # Vong lap chinh
    running = True
    simulation_time = 0
    paused = False
    speed_factor = 1
    max_cycle_time = max(cycle_times.values())

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed_factor = min(speed_factor + 0.5, 3)
                elif event.key == pygame.K_DOWN:
                    speed_factor = max(speed_factor - 0.5, 0.5)
                elif event.key == pygame.K_r:
                    simulation_time = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Cap nhat thoi gian mo phong
        if not paused:
            simulation_time = (simulation_time + 0.1 * speed_factor) % max_cycle_time

        # Xoa man hinh
        screen.fill((30, 30, 30))

        # Ve cac lien ket giua nut giao
        draw_connections()

        # Ve xe di chuyen giua cac nut
        draw_vehicles_on_connections(simulation_time)

        # Ve cac nut giao thong
        for intersection_name, position in intersection_positions.items():
            # Tinh thoi gian hien tai cho nut giao nay (da bao gom offset)
            intersection_time = (simulation_time - offset_times[intersection_name]) % cycle_times[intersection_name]

            # Ve nut giao
            draw_intersection(position, intersection_size, intersection_name, intersection_time)

            # Xac dinh va ve trang thai den
            for road in network.roads_per_intersection:
                road_id = f"{intersection_name}-{road}"

                # Kiem tra xem den co xanh khong
                is_green = False

                # Tim pha cua duong nay
                for phase_index, phase in enumerate(network.phase_groups[intersection_name]):
                    if road_id in phase:
                        # Tinh thoi gian bat dau va ket thuc cua pha
                        if phase_index == 0:
                            green_start = 0
                        else:
                            green_start = \
                            signal_timings[intersection_name][network.phase_groups[intersection_name][0][0]][
                                "green_time"] + network.amber_time

                        green_end = green_start + signal_timings[intersection_name][road_id]["green_time"]

                        # Kiem tra xem hien tai co trong thoi gian den xanh hay khong
                        if green_start <= intersection_time < green_end:
                            is_green = True
                            break

                # Ve den giao thong
                draw_traffic_light(traffic_light_positions[road_id], road_id, is_green)

        # Ve bang thong tin va huong dan
        draw_comparison_table()
        draw_instructions()

        # Ve thong tin mo phong
        text_time = font_medium.render(f"Thoi gian mo phong: {simulation_time:.1f}s", True, WHITE)
        text_speed = font_medium.render(f"Toc do: x{speed_factor:.1f}", True, WHITE)
        text_status = font_medium.render("Trang thai: " + ("Tam dung" if paused else "Dang chay"), True, WHITE)

        screen.blit(text_time, (20, 200))
        screen.blit(text_speed, (20, 230))
        screen.blit(text_status, (20, 260))

        # Cap nhat man hinh
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    print("=== HE THONG DIEU KHIEN MANG LUOI DEN TIN HIEU GIAO THONG THONG MINH ===")
    print("Su dung thuat toan Webster ket hop Logic mo va dong bo hoa mang luoi")
    print("Mo phong tren mang luoi da giac 6 nut giao")
    print("-----------------------------------------------------------")

    try:
        # Cac bo du lieu mat do tac nghen duoc dinh nghia san
        predefined_data_sets = {
            "1": {  # Truong hop mat do dong deu
                "Nut 1-A": 0.3, "Nut 1-B": 0.3, "Nut 1-C": 0.3, "Nut 1-D": 0.3,
                "Nut 2-A": 0.3, "Nut 2-B": 0.3, "Nut 2-C": 0.3, "Nut 2-D": 0.3,
                "Nut 3-A": 0.3, "Nut 3-B": 0.3, "Nut 3-C": 0.3, "Nut 3-D": 0.3,
                "Nut 4-A": 0.3, "Nut 4-B": 0.3, "Nut 4-C": 0.3, "Nut 4-D": 0.3,
                "Nut 5-A": 0.3, "Nut 5-B": 0.3, "Nut 5-C": 0.3, "Nut 5-D": 0.3,
                "Nut 6-A": 0.3, "Nut 6-B": 0.3, "Nut 6-C": 0.3, "Nut 6-D": 0.3
            },
            "2": {  # Truong hop mat do tang dan tu Nut 1 den Nut 6
                "Nut 1-A": 0.2, "Nut 1-B": 0.2, "Nut 1-C": 0.2, "Nut 1-D": 0.2,
                "Nut 2-A": 0.3, "Nut 2-B": 0.3, "Nut 2-C": 0.3, "Nut 2-D": 0.3,
                "Nut 3-A": 0.4, "Nut 3-B": 0.4, "Nut 3-C": 0.4, "Nut 3-D": 0.4,
                "Nut 4-A": 0.5, "Nut 4-B": 0.5, "Nut 4-C": 0.5, "Nut 4-D": 0.5,
                "Nut 5-A": 0.6, "Nut 5-B": 0.6, "Nut 5-C": 0.6, "Nut 5-D": 0.6,
                "Nut 6-A": 0.7, "Nut 6-B": 0.7, "Nut 6-C": 0.7, "Nut 6-D": 0.7
            },
            "3": {  # Truong hop duong ngang (A-C) dong duc hon duong doc (B-D)
                "Nut 1-A": 0.7, "Nut 1-B": 0.3, "Nut 1-C": 0.7, "Nut 1-D": 0.3,
                "Nut 2-A": 0.7, "Nut 2-B": 0.3, "Nut 2-C": 0.7, "Nut 2-D": 0.3,
                "Nut 3-A": 0.7, "Nut 3-B": 0.3, "Nut 3-C": 0.7, "Nut 3-D": 0.3,
                "Nut 4-A": 0.7, "Nut 4-B": 0.3, "Nut 4-C": 0.7, "Nut 4-D": 0.3,
                "Nut 5-A": 0.7, "Nut 5-B": 0.3, "Nut 5-C": 0.7, "Nut 5-D": 0.3,
                "Nut 6-A": 0.7, "Nut 6-B": 0.3, "Nut 6-C": 0.7, "Nut 6-D": 0.3
            },
            "4": {  # Truong hop duong doc (B-D) dong duc hon duong ngang (A-C)
                "Nut 1-A": 0.3, "Nut 1-B": 0.7, "Nut 1-C": 0.3, "Nut 1-D": 0.7,
                "Nut 2-A": 0.3, "Nut 2-B": 0.7, "Nut 2-C": 0.3, "Nut 2-D": 0.7,
                "Nut 3-A": 0.3, "Nut 3-B": 0.7, "Nut 3-C": 0.3, "Nut 3-D": 0.7,
                "Nut 4-A": 0.3, "Nut 4-B": 0.7, "Nut 4-C": 0.3, "Nut 4-D": 0.7,
                "Nut 5-A": 0.3, "Nut 5-B": 0.7, "Nut 5-C": 0.3, "Nut 5-D": 0.7,
                "Nut 6-A": 0.3, "Nut 6-B": 0.7, "Nut 6-C": 0.3, "Nut 6-D": 0.7
            },
            "5": {  # Truong hop co nhung nut giao dong duc (F va E - Nut 2 va Nut 5)
                "Nut 1-A": 0.3, "Nut 1-B": 0.3, "Nut 1-C": 0.3, "Nut 1-D": 0.3,
                "Nut 2-A": 0.8, "Nut 2-B": 0.8, "Nut 2-C": 0.8, "Nut 2-D": 0.8,
                "Nut 3-A": 0.3, "Nut 3-B": 0.3, "Nut 3-C": 0.3, "Nut 3-D": 0.3,
                "Nut 4-A": 0.3, "Nut 4-B": 0.3, "Nut 4-C": 0.3, "Nut 4-D": 0.3,
                "Nut 5-A": 0.8, "Nut 5-B": 0.8, "Nut 5-C": 0.8, "Nut 5-D": 0.8,
                "Nut 6-A": 0.3, "Nut 6-B": 0.3, "Nut 6-C": 0.3, "Nut 6-D": 0.3
            }
        }

        print("\nChon cach nhap du lieu mat do giao thong:")
        print("1. Su dung bo du lieu co san")
        print("2. Nhap thu cong")
        data_choice = input("Lua chon cua ban (1/2): ")

        congestion_data = {}

        if data_choice == "1":
            print("\nChon bo du lieu mat do giao thong:")
            print("1. Mat do dong deu (0.3)")
            print("2. Mat do tang dan tu Nut 1 den Nut 6")
            print("3. Duong ngang (A-C) dong duc hon duong doc (B-D)")
            print("4. Duong doc (B-D) dong duc hon duong ngang (A-C)")
            print("5. Mat do cao tai cac nut F va E (Nut 2 va Nut 5)")

            data_set_choice = input("Lua chon bo du lieu (1-5): ")

            if data_set_choice in predefined_data_sets:
                congestion_data = predefined_data_sets[data_set_choice]
                print("\nDu lieu chi so tac nghen da chon:")
                for intersection in range(1, 7):  # Updated to 6 intersections
                    print(f"Nut {intersection}:", end=" ")
                    for road in ["A", "B", "C", "D"]:
                        print(f"{road}={congestion_data[f'Nut {intersection}-{road}']:.2f}", end=" ")
                    print()
            else:
                print("Lua chon khong hop le! Su dung bo du lieu mac dinh (1).")
                congestion_data = predefined_data_sets["1"]

        elif data_choice == "2":
            print("\nNhap chi so tac nghen (0.01-1.00) cho cac duong tai moi nut giao:")
            for i in range(1, 7):  # Updated to 6 intersections
                print(f"\nNut giao {i}:")
                for road in ["A", "B", "C", "D"]:
                    value = float(input(f"  Chi so tac nghen duong {road}: "))
                    if not (0.01 <= value <= 1.00):
                        print("Loi: Chi so tac nghen phai nam trong khoang 0.01 den 1.00")
                        return
                    congestion_data[f"Nut {i}-{road}"] = value

            # Hien thi thong tin da nhap
            print("\nDu lieu chi so tac nghen da nhap:")
            for intersection in range(1, 7):  # Updated to 6 intersections
                print(f"Nut {intersection}:", end=" ")
                for road in ["A", "B", "C", "D"]:
                    print(f"{road}={congestion_data[f'Nut {intersection}-{road}']:.2f}", end=" ")
                print()

        else:
            print("Lua chon khong hop le! Su dung bo du lieu mac dinh (1).")
            congestion_data = predefined_data_sets["1"]

        # Tao mang luoi giao thong
        network = TrafficNetwork(congestion_data)

        # Hoi nguoi dung muon su dung che do nao
        print("\nChon phuong phap toi uu hoa:")
        print("1. Toi uu doc lap (moi nut giao duoc toi uu rieng biet)")
        print("2. Toi uu mang luoi (ca mang luoi duoc toi uu dong thoi)")
        choice = input("Lua chon cua ban (1/2): ")

        if choice == "1":
            signal_timings, cycle_times = network.optimize_independent()
            network.offset_times = {intersection: 0 for intersection in network.intersections}

            # Hien thi ket qua
            print("\nKet qua toi uu doc lap:")
            for intersection in network.intersections:
                print(f"\n{intersection}:")
                print(f"Chu ky: {cycle_times[intersection]} giay")
                for road in network.roads_per_intersection:
                    road_id = f"{intersection}-{road}"
                    timing = signal_timings[intersection][road_id]
                    print(
                        f"  Duong {road}: Xanh={timing['green_time']}s, Vang={timing['amber_time']}s, Do={timing['red_time']}s")

        elif choice == "2":
            signal_timings, cycle_times, offset_times = network.optimize_network()

            # Hien thi ket qua
            print("\nKet qua toi uu mang luoi:")
            for intersection in network.intersections:
                print(f"\n{intersection}:")
                print(f"Chu ky: {cycle_times[intersection]} giay")
                print(f"Offset: {offset_times[intersection]} giay")
                for road in network.roads_per_intersection:
                    road_id = f"{intersection}-{road}"
                    timing = signal_timings[intersection][road_id]
                    print(
                        f"  Duong {road}: Xanh={timing['green_time']}s, Vang={timing['amber_time']}s, Do={timing['red_time']}s")

        else:
            print("Lua chon khong hop le!")
            return

        # Hoi nguoi dung co muon chay mo phong
        print("\nBan co muon chay mo phong khong? (y/n)")
        if input().lower() == 'y':
            print("Dang khoi dong mo phong...")
            try:
                run_network_simulation(network)
            except Exception as e:
                print(f"Loi khi chay mo phong: {e}")
                print("Hay dam bao ban da cai dat thu vien pygame: pip install pygame")

    except ValueError:
        print("Loi: Vui long nhap gia tri so thuc hop le")
    except ImportError as e:
        print(f"Loi: {e}")
        print("Ban can cai dat cac thu vien sau de chay chuong trinh:")
        print("pip install numpy scikit-fuzzy pygame")


if __name__ == "__main__":
    main()