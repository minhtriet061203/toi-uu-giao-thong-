import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygame
import sys
import time
import math


class TrafficNetwork6Nodes:
    def __init__(self, congestion_data):
        """
        Khoi tao mang luoi giao thong voi 6 nut giao (A, B, C, D, E, F)
        congestion_data: dictionary chua chi so tac nghen cho moi duong tai moi nut giao
        """
        self.congestion_data = congestion_data
        self.intersections = ["A", "B", "C", "D", "E", "F"]
        self.roads_per_intersection = ["North", "South", "East", "West"]

        # Thong so co ban
        self.min_green_time = 15  # Thoi gian den xanh toi thieu (giay)
        self.max_green_time = 90  # Thoi gian den xanh toi da (giay)
        self.amber_time = 3  # Thoi gian den vang (giay)
        self.clearance_time = 2  # Thoi gian giai phong giao lo (giay)

        # Ma tran khoang cach giua cac nut giao (don vi: met)
        # Dua tren so do: A-B-C tren hang tren, F-E-D tren hang duoi
        self.distance_matrix = {
            "A": {"A": 0, "B": 300, "C": 600, "D": 0, "E": 0, "F": 400},
            "B": {"A": 300, "B": 0, "C": 300, "D": 0, "E": 350, "F": 0},
            "C": {"A": 600, "B": 300, "C": 0, "D": 400, "E": 0, "F": 0},
            "D": {"A": 0, "B": 0, "C": 400, "D": 0, "E": 300, "F": 0},
            "E": {"A": 0, "B": 350, "C": 0, "D": 300, "E": 0, "F": 300},
            "F": {"A": 400, "B": 0, "C": 0, "D": 0, "E": 300, "F": 0}
        }

        # Ma tran ket noi giua cac nut giao
        # Value: (duong ra tu nut 1, duong vao nut 2) hoac None neu khong ket noi truc tiep
        self.connections = {
            "A": {"B": ("East", "West"), "F": ("South", "North"), "C": None, "D": None, "E": None},
            "B": {"A": ("West", "East"), "C": ("East", "West"), "E": ("South", "North"), "D": None, "F": None},
            "C": {"B": ("West", "East"), "D": ("South", "North"), "A": None, "E": None, "F": None},
            "D": {"C": ("North", "South"), "E": ("West", "East"), "A": None, "B": None, "F": None},
            "E": {"D": ("East", "West"), "F": ("West", "East"), "B": ("North", "South"), "A": None, "C": None},
            "F": {"E": ("East", "West"), "A": ("North", "South"), "B": None, "C": None, "D": None}
        }

        # Thiet lap cac nhom pha cho moi nut giao
        # (duong North,South di chung mot pha va East,West di chung pha khac)
        self.phase_groups = {
            intersection: [
                [f"{intersection}-North", f"{intersection}-South"],
                [f"{intersection}-East", f"{intersection}-West"]
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
            connection = self.connections[reference_intersection].get(intersection)
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
                    green_start_time_dest = 0
                else:
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
                    green_start_time_source = 0
                else:
                    green_start_time_source = self.signal_timings[reference_intersection][phase_with_out_road[0]][
                                                  "green_time"] + self.amber_time

                # Tinh offset
                cycle_time = self.cycle_times[intersection]
                ideal_offset = (green_start_time_source + travel_time - green_start_time_dest) % cycle_time
                self.offset_times[intersection] = round(ideal_offset)

        return self.offset_times

    def normalize_cycle_times(self):
        """
        Dieu chinh chu ky cua cac nut giao thong de chung dong bo voi nhau
        """
        max_cycle = max(self.cycle_times.values())
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
                new_green_time = max(new_green_time, self.min_green_time)

                # Cap nhat thoi gian xanh moi
                for road in phase:
                    self.signal_timings[intersection][road]["green_time"] = new_green_time

            # Tinh lai thoi gian do cho moi pha
            for i, phase in enumerate(phase_groups):
                for road in phase:
                    red_time = cycle_time - self.signal_timings[intersection][road]["green_time"] - self.amber_time
                    self.signal_timings[intersection][road]["red_time"] = red_time

        return self.signal_timings

    def find_shortest_path(self, start, end):
        """
        Tim duong di ngan nhat giua hai nut giao thong bang thuat toan Dijkstra
        """
        distances = {intersection: float('infinity') for intersection in self.intersections}
        distances[start] = 0

        unvisited = list(self.intersections)
        previous = {intersection: None for intersection in self.intersections}

        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])

            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = previous[current]
                return path[::-1]

            if distances[current] == float('infinity'):
                break

            unvisited.remove(current)

            # Kiem tra cac nut ke
            for neighbor in self.intersections:
                if neighbor == current:
                    continue

                if self.connections[current].get(neighbor):
                    distance = self.distance_matrix[current][neighbor]
                    alternative_route = distances[current] + distance

                    if alternative_route < distances[neighbor]:
                        distances[neighbor] = alternative_route
                        previous[neighbor] = current

        return []


def run_network_simulation_6_nodes(network):
    """
    Mo phong truc quan hoat dong cua mang luoi den giao thong 6 nut
    """
    pygame.init()
    screen = pygame.display.set_mode((1400, 900))
    pygame.display.set_caption("Mo phong Mang luoi 6 Nut Giao thong Thong minh")
    clock = pygame.time.Clock()

    # Font chu
    font_small = pygame.font.Font(None, 20)
    font_medium = pygame.font.Font(None, 28)
    font_large = pygame.font.Font(None, 36)

    # Mau sac
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (150, 150, 150)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (173, 216, 230)
    DARK_GRAY = (50, 50, 50)

    # Vi tri cac nut giao thong theo so do A-B-C tren, F-E-D duoi
    intersection_positions = {
        "A": (250, 200),
        "B": (550, 200),
        "C": (850, 200),
        "F": (250, 500),
        "E": (550, 500),
        "D": (850, 500)
    }

    # Kich thuoc cua moi nut giao
    intersection_size = 100

    # Vi tri cac den giao thong cho moi nut giao
    traffic_light_positions = {}
    for intersection, center in intersection_positions.items():
        x, y = center
        traffic_light_positions[f"{intersection}-North"] = (x, y - intersection_size // 2)
        traffic_light_positions[f"{intersection}-South"] = (x, y + intersection_size // 2)
        traffic_light_positions[f"{intersection}-East"] = (x + intersection_size // 2, y)
        traffic_light_positions[f"{intersection}-West"] = (x - intersection_size // 2, y)

    # Lay thong tin tu network
    signal_timings = network.signal_timings
    cycle_times = network.cycle_times
    offset_times = network.offset_times
    congestion_data = network.congestion_data

    def draw_intersection(center, size, intersection_name, time_in_cycle):
        """Ve nut giao thong"""
        x, y = center

        # Ve hinh vuong dai dien cho nut giao
        pygame.draw.rect(screen, BLACK, (x - size // 2, y - size // 2, size, size))

        # Ve duong doc va ngang qua nut giao
        road_width = 25
        pygame.draw.rect(screen, GRAY, (x - size // 2, y - road_width // 2, size, road_width))  # Duong ngang
        pygame.draw.rect(screen, GRAY, (x - road_width // 2, y - size // 2, road_width, size))  # Duong doc

        # Ve vach ke duong
        line_width = 2
        line_length = 8
        line_gap = 8

        # Vach ke duong ngang
        for i in range(size // 2 // (line_length + line_gap)):
            pygame.draw.rect(screen, WHITE,
                             (x - size // 2 + i * (line_length + line_gap),
                              y - line_width // 2,
                              line_length, line_width))
            pygame.draw.rect(screen, WHITE,
                             (x + i * (line_length + line_gap),
                              y - line_width // 2,
                              line_length, line_width))

        # Vach ke duong doc
        for i in range(size // 2 // (line_length + line_gap)):
            pygame.draw.rect(screen, WHITE,
                             (x - line_width // 2,
                              y - size // 2 + i * (line_length + line_gap),
                              line_width, line_length))
            pygame.draw.rect(screen, WHITE,
                             (x - line_width // 2,
                              y + i * (line_length + line_gap),
                              line_width, line_length))

        # Ve ten nut giao
        text = font_large.render(intersection_name, True, WHITE)
        screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

        # Ve thong tin thoi gian
        cycle_time = cycle_times[intersection_name]
        offset = offset_times[intersection_name]
        adjusted_time = (time_in_cycle - offset) % cycle_time

        text_cycle = font_small.render(f"Chu ky: {cycle_time}s", True, WHITE)
        text_current = font_small.render(f"Thoi gian: {adjusted_time:.1f}s", True, WHITE)
        text_offset = font_small.render(f"Offset: {offset}s", True, WHITE)

        screen.blit(text_cycle, (x - size // 2, y + size // 2 + 5))
        screen.blit(text_current, (x - size // 2, y + size // 2 + 25))
        screen.blit(text_offset, (x - size // 2, y + size // 2 + 45))

    def draw_traffic_light(position, road_id, is_green):
        """Ve den giao thong"""
        x, y = position
        intersection, road = road_id.split("-")

        # Tim vi tri cua den dua tren duong
        if road == "North":
            light_x, light_y = x, y - 15
        elif road == "South":
            light_x, light_y = x, y + 15
        elif road == "East":
            light_x, light_y = x + 15, y
        elif road == "West":
            light_x, light_y = x - 15, y

        # Ve den giao thong
        color = GREEN if is_green else RED
        pygame.draw.circle(screen, color, (light_x, light_y), 6)

        # Ve ten duong
        text = font_small.render(road[0], True, WHITE)  # Chi hien thi chu cai dau
        if road == "North":
            screen.blit(text, (light_x - 5, light_y - 25))
        elif road == "South":
            screen.blit(text, (light_x - 5, light_y + 15))
        elif road == "East":
            screen.blit(text, (light_x + 15, light_y - 5))
        elif road == "West":
            screen.blit(text, (light_x - 25, light_y - 5))

    def draw_connections():
        """Ve duong lien ket giua cac nut giao"""
        for intersection1 in network.intersections:
            for intersection2 in network.intersections:
                if network.connections[intersection1].get(intersection2):
                    pos1 = intersection_positions[intersection1]
                    pos2 = intersection_positions[intersection2]

                    # Ve duong lien ket
                    pygame.draw.line(screen, LIGHT_BLUE, pos1, pos2, 3)

                    # Ve thong tin lien ket
                    out_road, in_road = network.connections[intersection1][intersection2]
                    mid_point = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                    
                    text = font_small.render(f"{out_road[0]}->{in_road[0]}", True, WHITE)
                    screen.blit(text, mid_point)

    def draw_vehicles_on_connections(simulation_time, speed_factor):
        """Ve xe di chuyen tren cac duong lien ket"""
        for intersection1 in network.intersections:
            for intersection2 in network.intersections:
                if network.connections[intersection1].get(intersection2):
                    pos1 = intersection_positions[intersection1]
                    pos2 = intersection_positions[intersection2]

                    out_road, in_road = network.connections[intersection1][intersection2]
                    out_road_key = f"{intersection1}-{out_road}"

                    # Kiem tra den co xanh khong
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
                        is_out_road_green = out_road_signal_time < signal_timings[intersection1][out_road_key]["green_time"]
                    else:
                        green_start = signal_timings[intersection1][network.phase_groups[intersection1][0][0]]["green_time"] + network.amber_time
                        is_out_road_green = green_start <= out_road_signal_time < green_start + signal_timings[intersection1][out_road_key]["green_time"]

                    # Chi ve xe khi den xanh
                    if is_out_road_green:
                        direction = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                        length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                        normalized = (direction[0] / length, direction[1] / length)

                        # Tinh so luong xe dua tren mat do tac nghen
                        congestion = congestion_data[out_road_key]
                        num_vehicles = int(congestion * 12)

                        # Ve cac xe tren duong noi
                        for k in range(num_vehicles):
                            progress = (simulation_time * speed_factor / 8 + k / num_vehicles) % 1
                            vehicle_x = pos1[0] + progress * direction[0]
                            vehicle_y = pos1[1] + progress * direction[1]

                            pygame.draw.rect(screen, YELLOW,
                                             (vehicle_x - 4, vehicle_y - 4, 8, 8))

    def draw_info_panel():
        """Ve bang thong tin"""
        x, y = 20, 20
        width, height = 380, 140

        pygame.draw.rect(screen, DARK_GRAY, (x, y, width, height), border_radius=5)

        text_title = font_medium.render("He thong 6 nut giao thong", True, WHITE)
        screen.blit(text_title, (x + 10, y + 10))

        # Thong tin chu ky
        avg_cycle = sum(cycle_times.values()) / len(cycle_times)
        max_cycle = max(cycle_times.values())

        text1 = font_small.render(f"Chu ky chung: {max_cycle}s", True, WHITE)
        text2 = font_small.render(f"Chu ky trung binh: {avg_cycle:.1f}s", True, WHITE)
        text3 = font_small.render(f"Tong offset: {sum(offset_times.values())}s", True, WHITE)
        text4 = font_small.render("Toi uu hoa lan song xanh", True, WHITE)

        screen.blit(text1, (x + 10, y + 40))
        screen.blit(text2, (x + 10, y + 60))
        screen.blit(text3, (x + 10, y + 80))
        screen.blit(text4, (x + 10, y + 100))

    def draw_instructions():
        """Ve huong dan su dung"""
        x, y = 1000, 20
        width, height = 380, 120

        pygame.draw.rect(screen, DARK_GRAY, (x, y, width, height), border_radius=5)

        text_title = font_medium.render("Huong dan su dung", True, WHITE)
        screen.blit(text_title, (x + 10, y + 10))

        text1 = font_small.render("SPACE: Tam dung/Tiep tuc", True, WHITE)
        text2 = font_small.render("UP/DOWN: Tang/Giam toc do", True, WHITE)
        text3 = font_small.render("R: Khoi dong lai", True, WHITE)
        text4 = font_small.render("ESC: Thoat", True, WHITE)

        screen.blit(text1, (x + 10, y + 35))
        screen.blit(text2, (x + 10, y + 55))
        screen.blit(text3, (x + 10, y + 75))
        screen.blit(text4, (x + 10, y + 95))

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

        # Ve cac ket noi
        draw_connections()

        # Ve xe di chuyen
        draw_vehicles_on_connections(simulation_time, speed_factor)

        # Ve cac nut giao thong
        for intersection_name, position in intersection_positions.items():
            intersection_time = (simulation_time - offset_times[intersection_name]) % cycle_times[intersection_name]
            draw_intersection(position, intersection_size, intersection_name, intersection_time)

            # Ve den giao thong
            for road in network.roads_per_intersection:
                road_id = f"{intersection_name}-{road}"
                is_green = False

                # Tim pha cua duong nay
                for phase_index, phase in enumerate(network.phase_groups[intersection_name]):
                    if road_id in phase:
                        if phase_index == 0:
                            green_start = 0
                        else:
                            green_start = signal_timings[intersection_name][network.phase_groups[intersection_name][0][0]]["green_time"] + network.amber_time

                        green_end = green_start + signal_timings[intersection_name][road_id]["green_time"]

                        if green_start <= intersection_time < green_end:
                            is_green = True
                            break

                draw_traffic_light(traffic_light_positions[road_id], road_id, is_green)

        # Ve bang thong tin
        draw_info_panel()
        draw_instructions()

        # Ve thong tin mo phong
        text_time = font_medium.render(f"Thoi gian: {simulation_time:.1f}s", True, WHITE)
        text_speed = font_medium.render(f"Toc do: x{speed_factor:.1f}", True, WHITE)
        text_status = font_medium.render("Trang thai: " + ("Tam dung" if paused else "Dang chay"), True, WHITE)

        screen.blit(text_time, (20, 180))
        screen.blit(text_speed, (20, 210))
        screen.blit(text_status, (20, 240))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    print("=== HE THONG DIEU KHIEN MANG LUOI 6 NUT GIAO THONG THONG MINH ===")
    print("Su dung thuat toan Webster ket hop Logic mo va dong bo hoa mang luoi")
    print("So do: A-B-C tren hang tren, F-E-D tren hang duoi")
    print("-----------------------------------------------------------")

    try:
        # Cac bo du lieu mat do tac nghen duoc dinh nghia san cho 6 nut
        predefined_data_sets = {
            "1": {  # Mat do dong deu
                **{f"{node}-{road}": 0.3 for node in ["A", "B", "C", "D", "E", "F"] 
                   for road in ["North", "South", "East", "West"]}
            },
            "2": {  # Mat do tang dan tu A den F
                **{f"A-{road}": 0.2 for road in ["North", "South", "East", "West"]},
                **{f"B-{road}": 0.3 for road in ["North", "South", "East", "West"]},
                **{f"C-{road}": 0.4 for road in ["North", "South", "East", "West"]},
                **{f"D-{road}": 0.6 for road in ["North", "South", "East", "West"]},
                **{f"E-{road}": 0.7 for road in ["North", "South", "East", "West"]},
                **{f"F-{road}": 0.8 for road in ["North", "South", "East", "West"]}
            },
            "3": {  # Duong ngang (East-West) dong duc hon duong doc (North-South)
                **{f"{node}-North": 0.3 for node in ["A", "B", "C", "D", "E", "F"]},
                **{f"{node}-South": 0.3 for node in ["A", "B", "C", "D", "E", "F"]},
                **{f"{node}-East": 0.7 for node in ["A", "B", "C", "D", "E", "F"]},
                **{f"{node}-West": 0.7 for node in ["A", "B", "C", "D", "E", "F"]}
            },
            "4": {  # Duong doc (North-South) dong duc hon duong ngang (East-West)
                **{f"{node}-North": 0.7 for node in ["A", "B", "C", "D", "E", "F"]},
                **{f"{node}-South": 0.7 for node in ["A", "B", "C", "D", "E", "F"]},
                **{f"{node}-East": 0.3 for node in ["A", "B", "C", "D", "E", "F"]},
                **{f"{node}-West": 0.3 for node in ["A", "B", "C", "D", "E", "F"]}
            },
            "5": {  # Nut B va E dong duc nhat (trung tam)
                **{f"A-{road}": 0.3 for road in ["North", "South", "East", "West"]},
                **{f"B-{road}": 0.8 for road in ["North", "South", "East", "West"]},
                **{f"C-{road}": 0.3 for road in ["North", "South", "East", "West"]},
                **{f"D-{road}": 0.3 for road in ["North", "South", "East", "West"]},
                **{f"E-{road}": 0.8 for road in ["North", "South", "East", "West"]},
                **{f"F-{road}": 0.3 for road in ["North", "South", "East", "West"]}
            },
            "6": {  # Truong hop ngau nhien thuc te
                "A-North": 0.4, "A-South": 0.6, "A-East": 0.5, "A-West": 0.3,
                "B-North": 0.7, "B-South": 0.8, "B-East": 0.6, "B-West": 0.7,
                "C-North": 0.3, "C-South": 0.4, "C-East": 0.2, "C-West": 0.5,
                "D-North": 0.5, "D-South": 0.4, "D-East": 0.6, "D-West": 0.3,
                "E-North": 0.8, "E-South": 0.7, "E-East": 0.9, "E-West": 0.6,
                "F-North": 0.2, "F-South": 0.3, "F-East": 0.4, "F-West": 0.3
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
            print("2. Mat do tang dan tu A den F")
            print("3. Duong ngang (East-West) dong duc hon duong doc")
            print("4. Duong doc (North-South) dong duc hon duong ngang")
            print("5. Nut trung tam (B, E) dong duc nhat")
            print("6. Truong hop ngau nhien thuc te")

            data_set_choice = input("Lua chon bo du lieu (1-6): ")

            if data_set_choice in predefined_data_sets:
                congestion_data = predefined_data_sets[data_set_choice]
                print("\nDu lieu chi so tac nghen da chon:")
                for intersection in ["A", "B", "C", "D", "E", "F"]:
                    print(f"Nut {intersection}:", end=" ")
                    for road in ["North", "South", "East", "West"]:
                        print(f"{road[0]}={congestion_data[f'{intersection}-{road}']:.1f}", end=" ")
                    print()
            else:
                print("Lua chon khong hop le! Su dung bo du lieu mac dinh.")
                congestion_data = predefined_data_sets["1"]

        elif data_choice == "2":
            print("\nNhap chi so tac nghen (0.01-1.00) cho cac duong tai moi nut giao:")
            for intersection in ["A", "B", "C", "D", "E", "F"]:
                print(f"\nNut giao {intersection}:")
                for road in ["North", "South", "East", "West"]:
                    while True:
                        try:
                            value = float(input(f"  Chi so tac nghen duong {road}: "))
                            if 0.01 <= value <= 1.00:
                                congestion_data[f"{intersection}-{road}"] = value
                                break
                            else:
                                print("Chi so phai nam trong khoang 0.01 den 1.00")
                        except ValueError:
                            print("Vui long nhap gia tri so thuc hop le")

            # Hien thi thong tin da nhap
            print("\nDu lieu chi so tac nghen da nhap:")
            for intersection in ["A", "B", "C", "D", "E", "F"]:
                print(f"Nut {intersection}:", end=" ")
                for road in ["North", "South", "East", "West"]:
                    print(f"{road[0]}={congestion_data[f'{intersection}-{road}']:.1f}", end=" ")
                print()

        else:
            print("Lua chon khong hop le! Su dung bo du lieu mac dinh.")
            congestion_data = predefined_data_sets["1"]

        # Tao mang luoi giao thong 6 nut
        network = TrafficNetwork6Nodes(congestion_data)

        # Hoi nguoi dung muon su dung che do nao
        print("\nChon phuong phap toi uu hoa:")
        print("1. Toi uu doc lap (moi nut giao duoc toi uu rieng biet)")
        print("2. Toi uu mang luoi (ca mang luoi duoc toi uu dong thoi)")
        choice = input("Lua chon cua ban (1/2): ")

        if choice == "1":
            signal_timings, cycle_times = network.optimize_independent()
            network.offset_times = {intersection: 0 for intersection in network.intersections}

            print("\nKet qua toi uu doc lap:")
            for intersection in network.intersections:
                print(f"\n=== Nut {intersection} ===")
                print(f"Chu ky: {cycle_times[intersection]} giay")
                for road in network.roads_per_intersection:
                    road_id = f"{intersection}-{road}"
                    timing = signal_timings[intersection][road_id]
                    print(f"  {road}: Xanh={timing['green_time']}s, Vang={timing['amber_time']}s, Do={timing['red_time']}s")

        elif choice == "2":
            signal_timings, cycle_times, offset_times = network.optimize_network()

            print("\nKet qua toi uu mang luoi:")
            for intersection in network.intersections:
                print(f"\n=== Nut {intersection} ===")
                print(f"Chu ky: {cycle_times[intersection]} giay")
                print(f"Offset: {offset_times[intersection]} giay")
                for road in network.roads_per_intersection:
                    road_id = f"{intersection}-{road}"
                    timing = signal_timings[intersection][road_id]
                    print(f"  {road}: Xanh={timing['green_time']}s, Vang={timing['amber_time']}s, Do={timing['red_time']}s")

        else:
            print("Lua chon khong hop le!")
            return

        # Hoi nguoi dung co muon chay mo phong
        print("\nBan co muon chay mo phong truc quan khong? (y/n)")
        if input().lower() == 'y':
            print("Dang khoi dong mo phong...")
            print("So do mang luoi: A-B-C (hang tren), F-E-D (hang duoi)")
            try:
                run_network_simulation_6_nodes(network)
            except Exception as e:
                print(f"Loi khi chay mo phong: {e}")
                print("Hay dam bao ban da cai dat thu vien pygame: pip install pygame")

    except Exception as e:
        print(f"Loi: {e}")
        print("Ban can cai dat cac thu vien sau de chay chuong trinh:")
        print("pip install numpy scikit-fuzzy pygame")


if __name__ == "__main__":
    main()