import cv2
import numpy as np
from ultralytics import YOLO
import math
from scipy.spatial.distance import cdist
import os
from datetime import datetime
from collections import deque, defaultdict
import json
import threading
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygame
import sys


class VideoProcessor:
    def __init__(self, duong_dan_model="yolov8l.pt"):
        self.duong_dan_model = duong_dan_model
        
        self.HE_SO_PCU = {
            "car": 1.0,
            "truck": 2.5, 
            "bus": 3.0,
            "motorbike": 0.4
        }
        
        self.CHIEU_DAI_DUONG_KM = 0.05
        self.SO_LAN_DUONG = 3
        
        self.TRONG_SO_W1 = 0.3
        self.TRONG_SO_W2 = 0.7
        self.MAT_DO_VAT_LY_TOI_DA = 120
        self.MAT_DO_PCU_TOI_DA = 200
        
        try:
            self.model = YOLO(duong_dan_model)
            self.model.overrides['verbose'] = False
            self.model.overrides['save'] = False
        except Exception as e:
            raise ValueError(f"Không thể tải model YOLO: {e}")
        
        self.cac_loai_xe = {"car", "truck", "bus", "motorbike"}
        self.ten_cac_lop = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]
        self.id_lop_xe = {i: ten for i, ten in enumerate(self.ten_cac_lop) if ten in self.cac_loai_xe}
        
    def process_single_video(self, duong_dan_video, giao_lo, huong):
        cap = cv2.VideoCapture(duong_dan_video)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {duong_dan_video}")
        
        cac_track = {}
        id_tiep_theo = 1
        nguong_tracking = 80
        max_mat_tich = 8
        
        xu_ly_moi_n_frame = 2
        nguong_tin_cay_phat_hien = 0.4
        nguong_nms = 0.5
        
        kich_thuoc_cua_so = 30
        lich_su_tci = deque(maxlen=kich_thuoc_cua_so)
        
        so_frame = 0
        so_frame_da_xu_ly = 0
        
        chieu_rong_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        chieu_cao_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tong_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        le_x = int(chieu_rong_frame * 0.15)
        le_y = int(chieu_cao_frame * 0.15)
        vung_duong = np.array([
            [le_x, le_y],
            [chieu_rong_frame - le_x, le_y], 
            [chieu_rong_frame - le_x, chieu_cao_frame - le_y],
            [le_x, chieu_cao_frame - le_y]
        ], np.int32)
        
        mat_na_roi = np.zeros((chieu_cao_frame, chieu_rong_frame), dtype=np.uint8)
        cv2.fillPoly(mat_na_roi, [vung_duong], 255)
        
        print(f"Đang xử lý {giao_lo}-{huong}: {os.path.basename(duong_dan_video)}")
        print(f"  Video: {chieu_rong_frame}x{chieu_cao_frame}, {tong_frame} frames")
        
        last_progress = 0
        
        while True:
            thanh_cong, frame = cap.read()
            if not thanh_cong:
                break
            
            so_frame += 1
            
            progress = int((so_frame / tong_frame) * 100)
            if progress != last_progress and progress % 10 == 0:
                print(f"  Tiến độ: {progress}%")
                last_progress = progress
            
            if so_frame % xu_ly_moi_n_frame != 0:
                continue
            
            so_frame_da_xu_ly += 1
            
            ket_qua = self.model(
                frame,
                conf=nguong_tin_cay_phat_hien,
                iou=nguong_nms,
                classes=list(self.id_lop_xe.keys()),
                verbose=False,
                stream=True
            )
            
            cac_phat_hien = []
            for r in ket_qua:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                    
                boxes = r.boxes
                if len(boxes.xyxy) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()
                    
                    tam_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
                    tam_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
                    
                    for i in range(len(cls)):
                        if cls[i] in self.id_lop_xe and conf[i] > nguong_tin_cay_phat_hien:
                            x, y = int(tam_x[i]), int(tam_y[i])
                            if 0 <= x < chieu_rong_frame and 0 <= y < chieu_cao_frame:
                                if mat_na_roi[y, x] > 0:
                                    ten_lop = self.ten_cac_lop[cls[i]]
                                    cac_phat_hien.append((tam_x[i], tam_y[i], ten_lop, conf[i]))
            
            if cac_phat_hien and cac_track:
                for phat_hien in cac_phat_hien:
                    tam_x, tam_y, ten_lop, tin_cay = phat_hien
                    
                    khoang_cach_min = float('inf')
                    id_gan_nhat = None
                    
                    for id_track, track in cac_track.items():
                        kc = math.sqrt((tam_x - track['tam_x'])**2 + (tam_y - track['tam_y'])**2)
                        if kc < khoang_cach_min and kc < nguong_tracking:
                            khoang_cach_min = kc
                            id_gan_nhat = id_track
                    
                    if id_gan_nhat:
                        cac_track[id_gan_nhat].update({
                            'tam_x': tam_x,
                            'tam_y': tam_y,
                            'lan_cuoi_nhin_thay': so_frame,
                            'mat_tich': 0
                        })
                    else:
                        pcu = self.HE_SO_PCU.get(ten_lop, 1.0)
                        cac_track[id_tiep_theo] = {
                            'id': id_tiep_theo,
                            'tam_x': tam_x,
                            'tam_y': tam_y,
                            'lop': ten_lop,
                            'pcu': pcu,
                            'lan_dau_nhin_thay': so_frame,
                            'lan_cuoi_nhin_thay': so_frame,
                            'mat_tich': 0
                        }
                        id_tiep_theo += 1
            elif cac_phat_hien and not cac_track:
                for phat_hien in cac_phat_hien:
                    tam_x, tam_y, ten_lop, tin_cay = phat_hien
                    pcu = self.HE_SO_PCU.get(ten_lop, 1.0)
                    cac_track[id_tiep_theo] = {
                        'id': id_tiep_theo,
                        'tam_x': tam_x,
                        'tam_y': tam_y,
                        'lop': ten_lop,
                        'pcu': pcu,
                        'lan_dau_nhin_thay': so_frame,
                        'lan_cuoi_nhin_thay': so_frame,
                        'mat_tich': 0
                    }
                    id_tiep_theo += 1
            
            danh_sach_xoa = []
            for id_track, track in cac_track.items():
                if track['lan_cuoi_nhin_thay'] < so_frame:
                    track['mat_tich'] += 1
                    if track['mat_tich'] > max_mat_tich:
                        danh_sach_xoa.append(id_track)
            
            for id_track in danh_sach_xoa:
                del cac_track[id_track]
            
            if cac_track:
                so_xe_hien_tai = len(cac_track)
                pcu_hien_tai = sum(track['pcu'] for track in cac_track.values())
                
                mat_do_vat_ly = so_xe_hien_tai / (self.CHIEU_DAI_DUONG_KM * self.SO_LAN_DUONG)
                mat_do_pcu = pcu_hien_tai / (self.CHIEU_DAI_DUONG_KM * self.SO_LAN_DUONG)
                
                chuan_hoa_vat_ly = min(mat_do_vat_ly / self.MAT_DO_VAT_LY_TOI_DA, 1.0)
                chuan_hoa_pcu = min(mat_do_pcu / self.MAT_DO_PCU_TOI_DA, 1.0)
                tci_hien_tai = self.TRONG_SO_W1 * chuan_hoa_vat_ly + self.TRONG_SO_W2 * chuan_hoa_pcu
                
                lich_su_tci.append(tci_hien_tai)
        
        cap.release()
        
        tci_tb = np.mean(lich_su_tci) if lich_su_tci else 0.3
        
        print(f"  ✓ Hoàn thành: TCI = {tci_tb:.3f}")
        print(f"  📊 Tổng xe đã theo dõi: {id_tiep_theo - 1}")
        
        return tci_tb


class TrafficOptimizer:
    def __init__(self, congestion_data):
        self.congestion_data = congestion_data
        self.intersections = ["A", "B", "C", "D", "E", "F"]
        self.roads_per_intersection = ["North", "South", "East", "West"]
        
        self.min_green_time = 15
        self.max_green_time = 90
        self.amber_time = 3
        self.clearance_time = 2
        
        self.distance_matrix = {
            "A": {"A": 0, "B": 300, "C": 600, "D": 0, "E": 0, "F": 400},
            "B": {"A": 300, "B": 0, "C": 300, "D": 0, "E": 350, "F": 0},
            "C": {"A": 600, "B": 300, "C": 0, "D": 400, "E": 0, "F": 0},
            "D": {"A": 0, "B": 0, "C": 400, "D": 0, "E": 300, "F": 0},
            "E": {"A": 0, "B": 350, "C": 0, "D": 300, "E": 0, "F": 300},
            "F": {"A": 400, "B": 0, "C": 0, "D": 0, "E": 300, "F": 0}
        }
        
        self.connections = {
            "A": {"B": ("East", "West"), "F": ("South", "North"), "C": None, "D": None, "E": None},
            "B": {"A": ("West", "East"), "C": ("East", "West"), "E": ("South", "North"), "D": None, "F": None},
            "C": {"B": ("West", "East"), "D": ("South", "North"), "A": None, "E": None, "F": None},
            "D": {"C": ("North", "South"), "E": ("West", "East"), "A": None, "B": None, "F": None},
            "E": {"D": ("East", "West"), "F": ("West", "East"), "B": ("North", "South"), "A": None, "C": None},
            "F": {"E": ("East", "West"), "A": ("North", "South"), "B": None, "C": None, "D": None}
        }
        
        self.phase_groups = {
            intersection: [
                [f"{intersection}-North", f"{intersection}-South"],
                [f"{intersection}-East", f"{intersection}-West"]
            ] for intersection in self.intersections
        }
        
        self.avg_speed = 8.33
        
        self.signal_timings = {}
        self.cycle_times = {}
        self.offset_times = {}
    
    def optimize_single_intersection(self, congestion_levels, intersection):
        phase_groups = self.phase_groups[intersection]
        
        phase_congestion = []
        for phase in phase_groups:
            max_congestion = max([congestion_levels[road] for road in phase])
            phase_congestion.append(max_congestion)
        
        congestion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'congestion')
        green_time = ctrl.Consequent(np.arange(self.min_green_time, self.max_green_time + 1, 1), 'green_time')
        
        congestion['low'] = fuzz.trimf(congestion.universe, [0, 0, 0.3])
        congestion['medium'] = fuzz.trimf(congestion.universe, [0.2, 0.5, 0.8])
        congestion['high'] = fuzz.trimf(congestion.universe, [0.7, 1, 1])
        
        green_time['short'] = fuzz.trimf(green_time.universe, [self.min_green_time, self.min_green_time, 35])
        green_time['medium'] = fuzz.trimf(green_time.universe, [30, 45, 60])
        green_time['long'] = fuzz.trimf(green_time.universe, [55, self.max_green_time, self.max_green_time])
        
        rule1 = ctrl.Rule(congestion['low'], green_time['short'])
        rule2 = ctrl.Rule(congestion['medium'], green_time['medium'])
        rule3 = ctrl.Rule(congestion['high'], green_time['long'])
        
        green_time_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        green_time_simulator = ctrl.ControlSystemSimulation(green_time_ctrl)
        
        lost_time_per_phase = self.amber_time + self.clearance_time
        total_lost_time = lost_time_per_phase * len(phase_groups)
        
        y_values = [min(0.95 * congestion, 0.95) for congestion in phase_congestion]
        Y = sum(y_values)
        
        optimal_cycle = (1.5 * total_lost_time + 5) / (1 - Y)
        optimal_cycle = min(max(optimal_cycle, 60), 150)
        
        effective_green_times = {}
        timings = {}
        
        for i, phase in enumerate(phase_groups):
            green_time_simulator.input['congestion'] = phase_congestion[i]
            green_time_simulator.compute()
            initial_green = green_time_simulator.output['green_time']
            
            effective_green = (y_values[i] / Y) * (optimal_cycle - total_lost_time)
            final_green = 0.7 * effective_green + 0.3 * initial_green
            final_green = min(max(final_green, self.min_green_time), self.max_green_time)
            effective_green_times[i] = round(final_green)
        
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
        
        cycle_time = sum([effective_green_times[i] + self.amber_time for i in range(len(phase_groups))]) + total_lost_time
        
        return timings, round(cycle_time)
    
    def optimize_network(self):
        for intersection in self.intersections:
            congestion_at_intersection = {}
            for road in self.roads_per_intersection:
                key = f"{intersection}-{road}"
                congestion_at_intersection[key] = self.congestion_data[key]
            
            timings, cycle_time = self.optimize_single_intersection(congestion_at_intersection, intersection)
            self.signal_timings[intersection] = timings
            self.cycle_times[intersection] = cycle_time
        
        self.calculate_offsets()
        self.normalize_cycle_times()
        self.adjust_green_times()
        
        return self.signal_timings, self.cycle_times, self.offset_times
    
    def calculate_offsets(self):
        total_congestion_per_intersection = {}
        for intersection in self.intersections:
            total = sum(self.congestion_data[f"{intersection}-{road}"] for road in self.roads_per_intersection)
            total_congestion_per_intersection[intersection] = total
        
        reference_intersection = max(total_congestion_per_intersection, key=total_congestion_per_intersection.get)
        self.offset_times = {reference_intersection: 0}
        
        for intersection in self.intersections:
            if intersection == reference_intersection:
                continue
            
            connection = self.connections[reference_intersection].get(intersection)
            if connection:
                distance = self.distance_matrix[reference_intersection][intersection]
                travel_time = distance / self.avg_speed
                cycle_time = self.cycle_times[intersection]
                offset = travel_time % cycle_time
                self.offset_times[intersection] = round(offset)
            else:
                self.offset_times[intersection] = 0
        
        return self.offset_times
    
    def normalize_cycle_times(self):
        max_cycle = max(self.cycle_times.values())
        for intersection in self.intersections:
            self.cycle_times[intersection] = max_cycle
        return self.cycle_times
    
    def adjust_green_times(self):
        for intersection in self.intersections:
            cycle_time = self.cycle_times[intersection]
            phase_groups = self.phase_groups[intersection]
            
            current_green_times = [self.signal_timings[intersection][phase[0]]["green_time"] for phase in phase_groups]
            total_current_green = sum(current_green_times)
            
            lost_time_per_phase = self.amber_time + self.clearance_time
            total_lost_time = lost_time_per_phase * len(phase_groups)
            
            available_green_time = cycle_time - total_lost_time
            adjustment_ratio = available_green_time / total_current_green
            
            for i, phase in enumerate(phase_groups):
                new_green_time = round(current_green_times[i] * adjustment_ratio)
                new_green_time = max(new_green_time, self.min_green_time)
                
                for road in phase:
                    self.signal_timings[intersection][road]["green_time"] = new_green_time
            
            for i, phase in enumerate(phase_groups):
                for road in phase:
                    red_time = cycle_time - self.signal_timings[intersection][road]["green_time"] - self.amber_time
                    self.signal_timings[intersection][road]["red_time"] = red_time
        
        return self.signal_timings


class TrafficSimulator:
    def __init__(self, network):
        self.network = network
        pygame.init()
        self.screen = pygame.display.set_mode((1400, 900))
        pygame.display.set_caption("Mô phỏng Mạng lưới 6 Nút Giao thông Thông minh")
        self.clock = pygame.time.Clock()
        
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (150, 150, 150)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.LIGHT_BLUE = (173, 216, 230)
        self.DARK_GRAY = (50, 50, 50)
        
        self.intersection_positions = {
            "A": (250, 200), "B": (550, 200), "C": (850, 200),
            "F": (250, 500), "E": (550, 500), "D": (850, 500)
        }
        
        self.intersection_size = 100
    
    def run(self):
        running = True
        simulation_time = 0
        paused = False
        speed_factor = 1
        max_cycle_time = max(self.network.cycle_times.values())
        
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
            
            if not paused:
                simulation_time = (simulation_time + 0.1 * speed_factor) % max_cycle_time
            
            self.screen.fill((30, 30, 30))
            self.draw_connections()
            self.draw_vehicles(simulation_time, speed_factor)
            
            for intersection_name, position in self.intersection_positions.items():
                intersection_time = (simulation_time - self.network.offset_times[intersection_name]) % self.network.cycle_times[intersection_name]
                self.draw_intersection(position, self.intersection_size, intersection_name, intersection_time)
                self.draw_traffic_lights(intersection_name, intersection_time)
            
            self.draw_info_panel()
            self.draw_status(simulation_time, speed_factor, paused)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
    
    def draw_intersection(self, center, size, name, time_in_cycle):
        x, y = center
        
        pygame.draw.rect(self.screen, self.BLACK, (x - size // 2, y - size // 2, size, size))
        
        road_width = 25
        pygame.draw.rect(self.screen, self.GRAY, (x - size // 2, y - road_width // 2, size, road_width))
        pygame.draw.rect(self.screen, self.GRAY, (x - road_width // 2, y - size // 2, road_width, size))
        
        text = self.font_large.render(name, True, self.WHITE)
        self.screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))
        
        cycle_time = self.network.cycle_times[name]
        offset = self.network.offset_times[name]
        
        text_cycle = self.font_small.render(f"Chu kỳ: {cycle_time}s", True, self.WHITE)
        text_current = self.font_small.render(f"Thời gian: {time_in_cycle:.1f}s", True, self.WHITE)
        text_offset = self.font_small.render(f"Offset: {offset}s", True, self.WHITE)
        
        self.screen.blit(text_cycle, (x - size // 2, y + size // 2 + 5))
        self.screen.blit(text_current, (x - size // 2, y + size // 2 + 25))
        self.screen.blit(text_offset, (x - size // 2, y + size // 2 + 45))
    
    def draw_traffic_lights(self, intersection, time_in_cycle):
        x, y = self.intersection_positions[intersection]
        
        for road in self.network.roads_per_intersection:
            road_id = f"{intersection}-{road}"
            is_green = False
            
            for phase_index, phase in enumerate(self.network.phase_groups[intersection]):
                if road_id in phase:
                    if phase_index == 0:
                        green_start = 0
                    else:
                        green_start = self.network.signal_timings[intersection][self.network.phase_groups[intersection][0][0]]["green_time"] + self.network.amber_time
                    
                    green_end = green_start + self.network.signal_timings[intersection][road_id]["green_time"]
                    
                    if green_start <= time_in_cycle < green_end:
                        is_green = True
                        break
            
            if road == "North":
                light_x, light_y = x, y - self.intersection_size // 2 - 15
            elif road == "South":
                light_x, light_y = x, y + self.intersection_size // 2 + 15
            elif road == "East":
                light_x, light_y = x + self.intersection_size // 2 + 15, y
            elif road == "West":
                light_x, light_y = x - self.intersection_size // 2 - 15, y
            
            color = self.GREEN if is_green else self.RED
            pygame.draw.circle(self.screen, color, (light_x, light_y), 6)
            
            text = self.font_small.render(road[0], True, self.WHITE)
            if road == "North":
                self.screen.blit(text, (light_x - 5, light_y - 25))
            elif road == "South":
                self.screen.blit(text, (light_x - 5, light_y + 15))
            elif road == "East":
                self.screen.blit(text, (light_x + 15, light_y - 5))
            elif road == "West":
                self.screen.blit(text, (light_x - 25, light_y - 5))
    
    def draw_connections(self):
        for intersection1 in self.network.intersections:
            for intersection2 in self.network.intersections:
                if self.network.connections[intersection1].get(intersection2):
                    pos1 = self.intersection_positions[intersection1]
                    pos2 = self.intersection_positions[intersection2]
                    pygame.draw.line(self.screen, self.LIGHT_BLUE, pos1, pos2, 3)
    
    def draw_vehicles(self, simulation_time, speed_factor):
        for intersection1 in self.network.intersections:
            for intersection2 in self.network.intersections:
                if self.network.connections[intersection1].get(intersection2):
                    pos1 = self.intersection_positions[intersection1]
                    pos2 = self.intersection_positions[intersection2]
                    
                    out_road, in_road = self.network.connections[intersection1][intersection2]
                    out_road_key = f"{intersection1}-{out_road}"
                    
                    out_road_signal_time = (simulation_time - self.network.offset_times[intersection1]) % self.network.cycle_times[intersection1]
                    is_out_road_green = False
                    
                    for phase in self.network.phase_groups[intersection1]:
                        if out_road_key in phase:
                            phase_index = self.network.phase_groups[intersection1].index(phase)
                            if phase_index == 0:
                                is_out_road_green = out_road_signal_time < self.network.signal_timings[intersection1][out_road_key]["green_time"]
                            else:
                                green_start = self.network.signal_timings[intersection1][self.network.phase_groups[intersection1][0][0]]["green_time"] + self.network.amber_time
                                is_out_road_green = green_start <= out_road_signal_time < green_start + self.network.signal_timings[intersection1][out_road_key]["green_time"]
                            break
                    
                    if is_out_road_green:
                        direction = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                        length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                        normalized = (direction[0] / length, direction[1] / length)
                        
                        congestion = self.network.congestion_data[out_road_key]
                        num_vehicles = int(congestion * 12)
                        
                        for k in range(num_vehicles):
                            progress = (simulation_time * speed_factor / 8 + k / num_vehicles) % 1
                            vehicle_x = pos1[0] + progress * direction[0]
                            vehicle_y = pos1[1] + progress * direction[1]
                            pygame.draw.rect(self.screen, self.YELLOW, (vehicle_x - 4, vehicle_y - 4, 8, 8))
    
    def draw_info_panel(self):
        x, y = 20, 20
        width, height = 380, 140
        
        pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, width, height), border_radius=5)
        
        text_title = self.font_medium.render("Hệ thống 6 nút giao thông", True, self.WHITE)
        self.screen.blit(text_title, (x + 10, y + 10))
        
        avg_cycle = sum(self.network.cycle_times.values()) / len(self.network.cycle_times)
        max_cycle = max(self.network.cycle_times.values())
        
        text1 = self.font_small.render(f"Chu kỳ chung: {max_cycle}s", True, self.WHITE)
        text2 = self.font_small.render(f"Chu kỳ trung bình: {avg_cycle:.1f}s", True, self.WHITE)
        text3 = self.font_small.render(f"Tổng offset: {sum(self.network.offset_times.values())}s", True, self.WHITE)
        text4 = self.font_small.render("Tối ưu hóa làn sóng xanh", True, self.WHITE)
        
        self.screen.blit(text1, (x + 10, y + 40))
        self.screen.blit(text2, (x + 10, y + 60))
        self.screen.blit(text3, (x + 10, y + 80))
        self.screen.blit(text4, (x + 10, y + 100))
        
        x2, y2 = 1000, 20
        pygame.draw.rect(self.screen, self.DARK_GRAY, (x2, y2, 380, 120), border_radius=5)
        
        text_title2 = self.font_medium.render("Hướng dẫn", True, self.WHITE)
        self.screen.blit(text_title2, (x2 + 10, y2 + 10))
        
        instructions = [
            "SPACE: Tạm dừng/Tiếp tục",
            "UP/DOWN: Tăng/Giảm tốc độ",
            "R: Khởi động lại",
            "ESC: Thoát"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, self.WHITE)
            self.screen.blit(text, (x2 + 10, y2 + 35 + i * 20))
    
    def draw_status(self, simulation_time, speed_factor, paused):
        text_time = self.font_medium.render(f"Thời gian: {simulation_time:.1f}s", True, self.WHITE)
        text_speed = self.font_medium.render(f"Tốc độ: x{speed_factor:.1f}", True, self.WHITE)
        text_status = self.font_medium.render("Trạng thái: " + ("Tạm dừng" if paused else "Đang chạy"), True, self.WHITE)
        
        self.screen.blit(text_time, (20, 180))
        self.screen.blit(text_speed, (20, 210))
        self.screen.blit(text_status, (20, 240))


class TrafficOptimizationSystem:
    def __init__(self, model_path="yolov8l.pt"):
        self.model_path = model_path
        self.video_processor = VideoProcessor(model_path)
        self.congestion_data = {}
        self.optimizer = None
        self.simulator = None
    
    def input_videos_manually(self):
        print("=== NHẬP DỮ LIỆU VIDEO CHO 6 NÚT GIAO THÔNG ===")
        print("Bạn sẽ nhập đường dẫn video cho từng nút giao và từng hướng")
        print("Nếu không có video cho hướng nào, nhấn Enter để bỏ qua")
        print("-" * 60)
        
        start_time = time.time()
        video_count = 0
        
        direction_vn_to_en = {
            "Bắc": "North",
            "Nam": "South", 
            "Đông": "East",
            "Tây": "West"
        }
        
        for intersection in ["A", "B", "C", "D", "E", "F"]:
            print(f"\n🚦 NÚT GIAO {intersection}:")
            print("-" * 40)
            
            for direction_vn, direction_en in direction_vn_to_en.items():
                while True:
                    video_path = input(f"  📹 Video hướng {direction_vn}: ").strip()
                    
                    if not video_path:
                        print(f"     ⚠️  Không có video, sử dụng TCI mặc định = 0.3")
                        self.congestion_data[f"{intersection}-{direction_en}"] = 0.3
                        break
                    
                    if os.path.exists(video_path):
                        file_name = os.path.basename(video_path)
                        confirm = input(f"     Xác nhận xử lý '{file_name}'? (y/n/Enter=y): ").strip().lower()
                        
                        if confirm in ['', 'y', 'yes']:
                            try:
                                print(f"     🔄 Đang xử lý...")
                                tci = self.video_processor.process_single_video(
                                    video_path, intersection, direction_vn
                                )
                                self.congestion_data[f"{intersection}-{direction_en}"] = tci
                                video_count += 1
                                print(f"     ✅ Hoàn thành! TCI = {tci:.3f}")
                                break
                                
                            except Exception as e:
                                print(f"     ❌ Lỗi xử lý video: {e}")
                                retry = input("     Thử lại? (y/n): ").strip().lower()
                                if retry != 'y':
                                    self.congestion_data[f"{intersection}-{direction_en}"] = 0.3
                                    print(f"     ⚠️  Sử dụng TCI mặc định = 0.3")
                                    break
                        else:
                            continue
                    else:
                        print(f"     ❌ File không tồn tại!")
                        retry = input("     Nhập lại? (y/n): ").strip().lower()
                        if retry != 'y':
                            self.congestion_data[f"{intersection}-{direction_en}"] = 0.3
                            print(f"     ⚠️  Sử dụng TCI mặc định = 0.3")
                            break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ HOÀN THÀNH NHẬP DỮ LIỆU")
        print(f"⏱️  Tổng thời gian: {total_time:.2f}s ({total_time/60:.1f} phút)")
        print(f"📊 Số video đã xử lý: {video_count}/24")
        
        self.display_congestion_summary()
        self.save_temp_data()
        
        return self.congestion_data
    
    def input_tci_values_manually(self):
        print("=== NHẬP GIÁ TRỊ TCI TRỰC TIẾP ===")
        print("Nhập giá trị TCI (0.0 - 1.0) cho từng đường")
        print("-" * 60)
        
        direction_vn_to_en = {
            "Bắc": "North",
            "Nam": "South", 
            "Đông": "East",
            "Tây": "West"
        }
        
        for intersection in ["A", "B", "C", "D", "E", "F"]:
            print(f"\n🚦 NÚT GIAO {intersection}:")
            
            for direction_vn, direction_en in direction_vn_to_en.items():
                while True:
                    try:
                        value = input(f"  TCI hướng {direction_vn} (0.0-1.0): ").strip()
                        if not value:
                            tci = 0.3
                            print(f"     ⚠️  Sử dụng giá trị mặc định = 0.3")
                        else:
                            tci = float(value)
                            if 0.0 <= tci <= 1.0:
                                print(f"     ✅ TCI = {tci:.3f}")
                            else:
                                print("     ❌ Giá trị phải từ 0.0 đến 1.0!")
                                continue
                        
                        self.congestion_data[f"{intersection}-{direction_en}"] = tci
                        break
                        
                    except ValueError:
                        print("     ❌ Vui lòng nhập số thực hợp lệ!")
        
        self.display_congestion_summary()
        self.save_temp_data()
        
        return self.congestion_data
    
    def load_saved_data(self):
        temp_file = "temp_congestion_data.json"
        
        if os.path.exists(temp_file):
            print("\n📂 Tìm thấy dữ liệu đã lưu trước đó!")
            use_saved = input("Sử dụng dữ liệu này? (y/n): ").strip().lower()
            
            if use_saved == 'y':
                with open(temp_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    self.congestion_data = saved_data['congestion_data']
                    
                print("✅ Đã tải dữ liệu thành công!")
                self.display_congestion_summary()
                return True
        
        return False
    
    def save_temp_data(self):
        temp_file = "temp_congestion_data.json"
        
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "congestion_data": self.congestion_data
        }
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Đã lưu dữ liệu tạm vào: {temp_file}")
    
    def optimize_and_simulate(self):
        if not self.congestion_data:
            print("❌ Chưa có dữ liệu TCI. Vui lòng nhập dữ liệu trước!")
            return None
        
        print("\n=== TỐI ƯU HÓA HỆ THỐNG GIAO THÔNG ===")
        
        self.optimizer = TrafficOptimizer(self.congestion_data)
        
        signal_timings, cycle_times, offset_times = self.optimizer.optimize_network()
        
        print("\n📊 KẾT QUẢ TỐI ƯU HÓA:")
        print("-" * 50)
        
        for intersection in self.optimizer.intersections:
            print(f"\n🚦 Nút {intersection}:")
            print(f"  Chu kỳ: {cycle_times[intersection]}s")
            print(f"  Offset: {offset_times[intersection]}s")
            
            for road in self.optimizer.roads_per_intersection:
                road_id = f"{intersection}-{road}"
                timing = signal_timings[intersection][road_id]
                congestion = self.congestion_data[road_id]
                
                print(f"  {road}: TCI={congestion:.3f}, Xanh={timing['green_time']}s, Đỏ={timing['red_time']}s")
        
        self.save_results(signal_timings, cycle_times, offset_times)
        
        print("\n🎮 Bạn có muốn chạy mô phỏng không? (y/n)")
        if input().lower() == 'y':
            print("🚀 Đang khởi động mô phỏng...")
            print("\nHƯỚNG DẪN ĐIỀU KHIỂN:")
            print("  SPACE: Tạm dừng/Tiếp tục")
            print("  ↑/↓: Tăng/Giảm tốc độ")
            print("  R: Khởi động lại")
            print("  ESC: Thoát")
            print("\nNhấn Enter để bắt đầu...")
            input()
            
            self.simulator = TrafficSimulator(self.optimizer)
            self.simulator.run()
        
        return signal_timings, cycle_times, offset_times
    
    def display_congestion_summary(self):
        print("\n📊 TÓM TẮT MẬT ĐỘ GIAO THÔNG:")
        print("-" * 50)
        
        all_tci_values = list(self.congestion_data.values())
        avg_tci = sum(all_tci_values) / len(all_tci_values) if all_tci_values else 0
        
        congestion_levels = {
            "🟢 Thông thoáng": 0,
            "🟡 Ùn tắc nhẹ": 0,
            "🟠 Ùn tắc TB": 0,
            "🔴 Ùn tắc nặng": 0
        }
        
        for intersection in ["A", "B", "C", "D", "E", "F"]:
            print(f"\n🚦 Giao lộ {intersection}:")
            
            total_tci = 0
            count = 0
            
            for direction in ["North", "South", "East", "West"]:
                key = f"{intersection}-{direction}"
                if key in self.congestion_data:
                    tci = self.congestion_data[key]
                    total_tci += tci
                    count += 1
                    
                    if tci <= 0.3:
                        level = "🟢 Thông thoáng"
                        congestion_levels["🟢 Thông thoáng"] += 1
                    elif tci <= 0.6:
                        level = "🟡 Ùn tắc nhẹ"
                        congestion_levels["🟡 Ùn tắc nhẹ"] += 1
                    elif tci <= 0.8:
                        level = "🟠 Ùn tắc TB"
                        congestion_levels["🟠 Ùn tắc TB"] += 1
                    else:
                        level = "🔴 Ùn tắc nặng"
                        congestion_levels["🔴 Ùn tắc nặng"] += 1
                    
                    print(f"  {direction}: {tci:.3f} - {level}")
            
            if count > 0:
                avg_tci_intersection = total_tci / count
                print(f"  📈 Trung bình: {avg_tci_intersection:.3f}")
        
        print(f"\n{'='*50}")
        print("📊 THỐNG KÊ TỔNG QUAN:")
        print(f"  TCI trung bình toàn mạng: {avg_tci:.3f}")
        print(f"  TCI cao nhất: {max(all_tci_values):.3f}")
        print(f"  TCI thấp nhất: {min(all_tci_values):.3f}")
        
        print("\n📈 PHÂN BỐ MỨC ĐỘ ÙN TẮC:")
        total_roads = sum(congestion_levels.values())
        for level, count in congestion_levels.items():
            percentage = (count / total_roads * 100) if total_roads > 0 else 0
            print(f"  {level}: {count} đường ({percentage:.1f}%)")
    
    def save_results(self, signal_timings, cycle_times, offset_times):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "timestamp": timestamp,
            "congestion_data": self.congestion_data,
            "signal_timings": signal_timings,
            "cycle_times": cycle_times,
            "offset_times": offset_times
        }
        
        json_file = f"traffic_optimization_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Đã lưu kết quả vào: {json_file}")
        
        txt_file = f"traffic_optimization_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=== KẾT QUẢ TỐI ƯU HÓA HỆ THỐNG GIAO THÔNG 6 NÚT ===\n\n")
            f.write(f"Thời gian: {timestamp}\n\n")
            
            for intersection in self.optimizer.intersections:
                f.write(f"=== Nút {intersection} ===\n")
                f.write(f"Chu kỳ: {cycle_times[intersection]}s\n")
                f.write(f"Offset: {offset_times[intersection]}s\n\n")
                
                for road in self.optimizer.roads_per_intersection:
                    road_id = f"{intersection}-{road}"
                    timing = signal_timings[intersection][road_id]
                    congestion = self.congestion_data[road_id]
                    
                    f.write(f"{road}:\n")
                    f.write(f"  - TCI: {congestion:.3f}\n")
                    f.write(f"  - Đèn xanh: {timing['green_time']}s\n")
                    f.write(f"  - Đèn đỏ: {timing['red_time']}s\n")
                    f.write(f"  - Đèn vàng: {timing['amber_time']}s\n\n")
        
        print(f"✅ Đã lưu báo cáo vào: {txt_file}")


def main():
    print("🚦 === HỆ THỐNG XỬ LÝ VIDEO VÀ TỐI ƯU GIAO THÔNG 6 NÚT ===")
    print("=" * 60)
    
    system = TrafficOptimizationSystem()
    
    while True:
        print("\n📋 MENU CHỨC NĂNG:")
        print("1. 📹 Nhập video từng nút từng hướng")
        print("2. 📊 Nhập trực tiếp giá trị TCI")
        print("3. 📂 Sử dụng dữ liệu đã lưu")
        print("4. 🚀 Tối ưu hóa và mô phỏng")
        print("5. 📈 Xem tóm tắt dữ liệu hiện tại")
        print("6. 🚪 Thoát")
        
        choice = input("\nChọn chức năng (1-6): ").strip()
        
        if choice == "1":
            system.input_videos_manually()
        elif choice == "2":
            system.input_tci_values_manually()
        elif choice == "3":
            if not system.load_saved_data():
                print("❌ Không có dữ liệu đã lưu!")
        elif choice == "4":
            if system.congestion_data:
                system.optimize_and_simulate()
            else:
                print("❌ Chưa có dữ liệu! Vui lòng nhập dữ liệu trước.")
        elif choice == "5":
            if system.congestion_data:
                system.display_congestion_summary()
            else:
                print("❌ Chưa có dữ liệu để hiển thị!")
        elif choice == "6":
            print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()