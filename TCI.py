import cv2
import numpy as np
from ultralytics import YOLO
import math
from scipy.spatial import distance

class VehicleDensityTracker:
    def __init__(self, video_path, model_path):
        # Hệ số quy đổi phương tiện (PCU - Passenger Car Unit)
        self.PCU_FACTORS = {
            "car": 1.0,
            "truck": 2.5,
            "bus": 3.0,
            "motorbike": 0.4
        }

        # Thông số đường
        self.ROAD_LENGTH_KM = 0.05  # 50m -> 0.05km
        self.LANE_WIDTH = 3.5  # mét
        self.LANES = 3  # số làn đường

        # TCI parameters
        self.W1 = 0.3  # Weight for physical density
        self.W2 = 0.7  # Weight for PCU density
        self.MAX_PHYSICAL_DENSITY = 120  # Maximum reference physical density
        self.MAX_PCU_DENSITY = 200  # Maximum reference PCU density

        # Khởi tạo video và mô hình
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)

        # Danh sách các lớp xe
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

        # Vùng quan sát (polygon)
        self.road_area = self.define_road_area()

        # Thống kê
        self.frame_count = 0
        self.total_vehicles = 0
        self.total_pcu = 0
        
        # Tracking parameters
        self.tracked_vehicles = {}  # Dictionary để lưu thông tin xe đã track
        self.next_id = 1  # ID tiếp theo sẽ được gán cho xe mới
        self.tracking_threshold = 50  # Ngưỡng khoảng cách để xác định cùng một xe
        self.vehicle_history = []  # Lưu thông tin tất cả các xe đã đi qua
        self.active_tracks = {}  # Lưu các xe đang active trong vùng quan sát
        
        # Cumulative metrics
        self.cumulative_physical_density = 0
        self.cumulative_pcu_density = 0
        self.cumulative_tci = 0

    def define_road_area(self):
        """
        Xác định vùng quan sát của đoạn đường
        Hàm này có thể được điều chỉnh dựa trên video cụ thể
        """
        # Ví dụ: Tạo một vùng hình tứ giác làm vùng quan sát
        road_area = np.array([
            [200, 100],   # Điểm 1
            [600, 100],   # Điểm 2
            [600, 300],   # Điểm 3
            [200, 300]    # Điểm 4
        ], np.int32)
        return road_area
    
    def match_detections_to_tracks(self, detections):
        """
        Khớp các đối tượng phát hiện với các track hiện tại
        """
        new_active_tracks = {}
        
        # Lấy thông tin tất cả các xe đang được theo dõi
        track_centers = {}
        for track_id, track_info in self.active_tracks.items():
            track_centers[track_id] = (track_info['cx'], track_info['cy'])
        
        # Đối với mỗi phát hiện mới
        for detection in detections:
            cx, cy, class_name, conf = detection
            
            # Kiểm tra xem xe có trong vùng quan sát không
            if cv2.pointPolygonTest(self.road_area, (cx, cy), False) < 0:
                continue
                
            matched = False
            best_match_id = None
            min_distance = float('inf')
            
            # Tìm track gần nhất
            for track_id, (track_cx, track_cy) in track_centers.items():
                dist = distance.euclidean((cx, cy), (track_cx, track_cy))
                if dist < self.tracking_threshold and dist < min_distance:
                    min_distance = dist
                    best_match_id = track_id
                    matched = True
            
            if matched:
                # Cập nhật track hiện có
                self.active_tracks[best_match_id]['cx'] = cx
                self.active_tracks[best_match_id]['cy'] = cy
                self.active_tracks[best_match_id]['last_seen'] = self.frame_count
                new_active_tracks[best_match_id] = self.active_tracks[best_match_id]
                
                # Xóa khỏi danh sách track_centers để không khớp lại
                if best_match_id in track_centers:
                    del track_centers[best_match_id]
            else:
                # Tạo track mới
                pcu = self.PCU_FACTORS.get(class_name, 1.0)
                self.total_vehicles += 1
                self.total_pcu += pcu
                
                new_track = {
                    'id': self.next_id,
                    'cx': cx,
                    'cy': cy,
                    'class': class_name,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'pcu': pcu
                }
                
                new_active_tracks[self.next_id] = new_track
                self.vehicle_history.append(new_track)
                self.next_id += 1
        
        # Lọc ra các track không còn hoạt động (mất dấu quá lâu)
        for track_id, track_info in self.active_tracks.items():
            if (self.frame_count - track_info['last_seen']) <= 5:  # Giữ lại trong 5 frame
                if track_id not in new_active_tracks:
                    new_active_tracks[track_id] = track_info
        
        self.active_tracks = new_active_tracks
        return len(self.active_tracks)

    def calculate_tci(self, physical_density, pcu_density):
        """
        Tính toán chỉ số TCI (Traffic Congestion Index) dựa trên công thức:
        TCI = w1 * (Physical Density / Max Physical Density) + w2 * (PCU Density / Max PCU Density)
        """
        normalized_physical = min(physical_density / self.MAX_PHYSICAL_DENSITY, 1.0)
        normalized_pcu = min(pcu_density / self.MAX_PCU_DENSITY, 1.0)
        
        tci = self.W1 * normalized_physical + self.W2 * normalized_pcu
        return tci

    def get_congestion_level(self, tci):
        """
        Phân loại mức độ ùn tắc dựa trên giá trị TCI
        """
        if tci <= 0.3:
            return "Thong thoang", (0, 255, 0)  # Green
        elif tci <= 0.6:
            return "un tac nhe", (0, 255, 255)  # Yellow
        elif tci <= 0.8:
            return "un tac trung binh", (0, 165, 255)  # Orange
        else:
            return "un tac nghiem trong ", (0, 0, 255)  # Red

    def process_video(self):
        while True:
            # Đọc frame
            success, img = self.cap.read()
            if not success:
                break

            # Tăng biến đếm frame
            self.frame_count += 1

            # Detect các đối tượng
            results = self.model(img, stream=True)

            # Chuẩn bị danh sách các phát hiện cho frame hiện tại
            current_detections = []

            # Xử lý các đối tượng được detect
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Thông tin bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Tâm của đối tượng
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Lấy tên lớp và độ tin cậy
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    current_class = self.classNames[cls]

                    # Kiểm tra xem có phải là xe và đủ độ tin cậy không
                    if current_class in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                        current_detections.append((cx, cy, current_class, conf))
                        
                        # Vẽ bounding box cho xe
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(img, f"{current_class} {conf:.2f}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Khớp các phát hiện với các track
            active_vehicle_count = self.match_detections_to_tracks(current_detections)
            
            # Tính PCU của các xe đang ở trong frame hiện tại
            current_pcu = sum(track['pcu'] for track in self.active_tracks.values())

            # Tính mật độ cho frame hiện tại
            physical_density = active_vehicle_count / (self.ROAD_LENGTH_KM * self.LANES)
            pcu_density = current_pcu / (self.ROAD_LENGTH_KM * self.LANES)
            
            # Tính TCI cho frame hiện tại
            current_tci = self.calculate_tci(physical_density, pcu_density)
            congestion_level, color = self.get_congestion_level(current_tci)
            
            # Cập nhật mật độ tích lũy
            self.cumulative_physical_density += physical_density
            self.cumulative_pcu_density += pcu_density
            self.cumulative_tci += current_tci
            
            # Tính mật độ trung bình
            avg_physical_density = self.cumulative_physical_density / self.frame_count
            avg_pcu_density = self.cumulative_pcu_density / self.frame_count
            avg_tci = self.cumulative_tci / self.frame_count

            # Vẽ vùng quan sát
            cv2.polylines(img, [self.road_area], True, (0, 255, 0), 2)

            # Hiển thị thông tin
            cv2.putText(img, f"Active Vehicles: {active_vehicle_count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(img, f"Total Unique Vehicles: {self.total_vehicles}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(img, f"Physical Density: {physical_density:.1f} veh/km/lane", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(img, f"PCU Density: {pcu_density:.1f} pcu/km/lane", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
            cv2.putText(img, f"TCI: {current_tci:.2f}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, f"Congestion Level: {congestion_level}", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, f"Avg TCI: {avg_tci:.2f}", (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Visualize active tracks with IDs
            for track_id, track_info in self.active_tracks.items():
                cv2.putText(img, f"ID:{track_id}", (int(track_info['cx']), int(track_info['cy'])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # Hiển thị frame
            cv2.imshow("Vehicle Density & TCI Tracking", img)

            # Thoát nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Tổng kết sau khi xử lý video
        print("\n===== SUMMARY =====")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Unique Vehicles Detected: {self.total_vehicles}")
        print(f"Total PCU: {self.total_pcu:.2f}")
        print(f"Average Physical Density: {avg_physical_density:.2f} veh/km/lane")
        print(f"Average PCU Density: {avg_pcu_density:.2f} pcu/km/lane")
        print(f"Average TCI: {avg_tci:.2f}")
        print(f"Congestion Level: {self.get_congestion_level(avg_tci)[0]}")
        
        # Giải phóng tài nguyên
        self.cap.release()
        cv2.destroyAllWindows()

# Sử dụng
tracker = VehicleDensityTracker(
    video_path = r"C:\Users\Administrator\Downloads\Nut A\East.MOV",
    model_path="../Yolo-Weights/yolov8l.pt"
)
tracker.process_video()