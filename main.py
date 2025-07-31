import cv2
import numpy as np
import math
from kociemba import solve

class RubikCubeScanner:
    def __init__(self):
        # 魔術方塊面順序: URFDLB (Up, Right, Front, Down, Left, Back)
        self.faces = ['U', 'R', 'F', 'D', 'L', 'B']
        self.current_face = 0
        self.face_colors = {face: [''] * 9 for face in self.faces}
        self.scanning_mode = False
        self.scan_complete = False
        
        # 顏色校準參數
        self.color_calibration = {
            'white': {'s_max': 100, 'v_min': 180},
            'red': {'h_min': 0, 'h_max': 5, 'h_min2': 170, 'h_max2': 179, 's_min': 80, 'v_min': 80},
            'orange': {'h_min': 6, 'h_max': 20, 's_min': 100, 'v_min': 100},
            'yellow': {'h_min': 21, 'h_max': 40, 's_min': 120, 'v_min': 120},
            'green': {'h_min': 50, 'h_max': 70, 's_min': 80, 'v_min': 80},
            'blue': {'h_min': 91, 'h_max': 130, 's_min': 80, 'v_min': 80}
        }
        
        # 顏色對應到Kociemba格式
        self.color_mapping = {
            'white': 'U',
            'red': 'R', 
            'blue': 'B',
            'orange': 'L',
            'green': 'F',
            'yellow': 'D'
        }

    def classify_color(self, h, s, v):
        """顏色分類"""
        cal = self.color_calibration
        
        if s < cal['white']['s_max'] and v > cal['white']['v_min']:
            return "white"
        
        elif (((h >= cal['red']['h_min'] and h <= cal['red']['h_max']) or 
               (h >= cal['red']['h_min2'] and h <= cal['red']['h_max2'])) and 
              s > cal['red']['s_min'] and v > cal['red']['v_min']):
            return "red"
        
        elif (h >= cal['orange']['h_min'] and h <= cal['orange']['h_max'] and 
              s > cal['orange']['s_min'] and v > cal['orange']['v_min']):
            return "orange"
        
        elif (h >= cal['yellow']['h_min'] and h <= cal['yellow']['h_max'] and 
              s > cal['yellow']['s_min'] and v > cal['yellow']['v_min']):
            return "yellow"
        
        elif (h >= cal['green']['h_min'] and h <= cal['green']['h_max'] and 
              s > cal['green']['s_min'] and v > cal['green']['v_min']):
            return "green"
        
        elif (h >= cal['blue']['h_min'] and h <= cal['blue']['h_max'] and 
              s > cal['blue']['s_min'] and v > cal['blue']['v_min']):
            return "blue"
        else:
            return "unknown"

    def is_square(self, approx, contour_area):
        """判斷是否為方形"""
        if len(approx) != 4:
            return False
        if contour_area < 1000 or contour_area > 15000:
            return False

        # 邊長比篩選
        sides = []
        for i in range(4):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % 4][0]
            side_length = np.linalg.norm(pt1 - pt2)
            sides.append(side_length)

        max_side = max(sides)
        min_side = min(sides)
        if max_side / min_side > 1.5:
            return False

        if not cv2.isContourConvex(approx):
            return False

        # 檢查角度是否接近 90 度
        def angle(pt1, pt2, pt0):
            dx1 = pt1[0] - pt0[0]
            dy1 = pt1[1] - pt0[1]
            dx2 = pt2[0] - pt0[0]
            dy2 = pt2[1] - pt0[1]
            inner = dx1 * dx2 + dy1 * dy2
            norm = math.hypot(dx1, dy1) * math.hypot(dx2, dy2)
            # 防止除以零的錯誤，加上一個很小的數
            cosine = inner / (norm + 1e-10)
            # 使用 arccos 取得夾角（弧度），再轉成角度
            return np.arccos(cosine) * 180 / np.pi

        angles = []
        for i in range(4):
            angle_deg = angle(approx[(i - 1) % 4][0], approx[(i + 1) % 4][0], approx[i][0])
            angles.append(angle_deg)

        # 檢查所有角度是否都在 70-110 度範圍內
        if not all(70 <= a <= 110 for a in angles):
            return False

        return True

    def get_square_centers(self, square_contours):
        """獲取方塊的中心點並排序"""
        centers = []
        for approx in square_contours:
            x, y, w, h = cv2.boundingRect(approx)
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y, approx))
        
        # 按照3x3網格排序
        if len(centers) >= 9:
            # 找到邊界
            x_coords = [c[0] for c in centers]
            y_coords = [c[1] for c in centers]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # 分成3x3網格
            grid_width = (max_x - min_x) / 3
            grid_height = (max_y - min_y) / 3
            
            sorted_centers = []
            for row in range(3):
                for col in range(3):
                    target_x = min_x + col * grid_width + grid_width / 2
                    target_y = min_y + row * grid_height + grid_height / 2
                    
                    # 找到最近的方塊
                    closest = min(centers, key=lambda c: 
                        abs(c[0] - target_x) + abs(c[1] - target_y))
                    sorted_centers.append(closest)
                    centers.remove(closest)
            
            return sorted_centers
        return []

    def scan_current_face(self, frame):
        """掃描當前面的9個方塊顏色"""
        # 灰階 + CLAHE 對比增強
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # 模糊 + 邊緣
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 60)
        dilated = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        square_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

            if self.is_square(approx, area):
                square_contours.append(approx)

        # 獲取排序後的方塊中心
        sorted_squares = self.get_square_centers(square_contours)
        
        if len(sorted_squares) >= 9 and self.current_face < len(self.faces):
            current_face = self.faces[self.current_face]
            colors = []
            
            for i, (cx, cy, approx) in enumerate(sorted_squares[:9]):
                x, y, w, h = cv2.boundingRect(approx)
                margin = int(min(w, h) * 0.2)
                roi = frame[y+margin:y+h-margin, x+margin:x+w-margin]

                if roi.size > 0:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    avg_h = np.mean(hsv_roi[:, :, 0])
                    avg_s = np.mean(hsv_roi[:, :, 1])
                    avg_v = np.mean(hsv_roi[:, :, 2])

                    color_name = self.classify_color(avg_h, avg_s, avg_v)
                    colors.append(color_name)
                    
                    # 在畫面上標示
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    # cv2.putText(影像, 文字, 座標, 字型, 字元大小, 顏色, 線條種類)
                    cv2.putText(frame, f"{i+1}:{color_name}", (cx-20, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 更新當前面的顏色
            if len(colors) == 9:
                self.face_colors[current_face] = colors
                return True
        
        return False

    def get_kociemba_string(self):
        """將掃描結果轉換為Kociemba格式"""
        kociemba_string = ""
        
        # 按照URFDLB順序構建字符串
        for face in self.faces:
            colors = self.face_colors[face]
            for color in colors:
                if color in self.color_mapping:
                    kociemba_string += self.color_mapping[color]
                else:
                    kociemba_string += 'U'  # 默認值
        
        return kociemba_string

    def solve_cube(self):
        """使用Kociemba求解魔術方塊"""
        try:
            kociemba_string = self.get_kociemba_string()
            print(f"Kociemba string: {kociemba_string}")
            solution = solve(kociemba_string)
            return solution
        except Exception as e:
            print(f"求解錯誤: {e}")
            return None

    def run(self):
        """主運行函數"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("無法開啟攝影機")
            return

        print("魔術方塊掃描器啟動")
        print("按 's' 開始掃描模式")
        print("按 'n' 掃描下一面")
        print("按 'r' 求解魔術方塊")
        print("按 'q' 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像")
                break

            # 顯示當前狀態
            current_face = self.faces[self.current_face] if self.current_face < len(self.faces) else "Complete"
            cv2.putText(frame, f"Current Face: {current_face}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.scanning_mode:
                cv2.putText(frame, "Scanning Mode - Press 'n' to confirm", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 嘗試掃描當前面
                if self.scan_current_face(frame):
                    cv2.putText(frame, "9 squares detected!", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 顯示已掃描的面
            y_offset = 150
            for i, face in enumerate(self.faces):
                colors = self.face_colors[face]
                status = "OK" if all(c != '' for c in colors) else "X"
                cv2.putText(frame, f"{face}: {status}", (10, y_offset + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 在掃描模式下顯示當前檢測到的顏色
            if self.scanning_mode:
                # 獲取當前檢測到的方塊
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_eq = clahe.apply(gray)
                blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
                edged = cv2.Canny(blurred, 30, 60)
                dilated = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=3)
                contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                square_contours = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
                    if self.is_square(approx, area):
                        square_contours.append(approx)
                
                sorted_squares = self.get_square_centers(square_contours)
                
                cv2.putText(frame, "Detected Colors:", (10, y_offset + 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if len(sorted_squares) > 0:
                    # 顯示檢測到的顏色
                    for i, (cx, cy, approx) in enumerate(sorted_squares):
                        x, y, w, h = cv2.boundingRect(approx)
                        margin = int(min(w, h) * 0.2)
                        roi = frame[y+margin:y+h-margin, x+margin:x+w-margin]
                        
                        if roi.size > 0:
                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            avg_h = np.mean(hsv_roi[:, :, 0])
                            avg_s = np.mean(hsv_roi[:, :, 1])
                            avg_v = np.mean(hsv_roi[:, :, 2])
                            
                            color_name = self.classify_color(avg_h, avg_s, avg_v)
                            cv2.putText(frame, f"Square{i+1}: {color_name} (H:{int(avg_h)} S:{int(avg_s)} V:{int(avg_v)})", 
                                        (10, y_offset + 225 + i * 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Rubik Cube Scanner", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.scanning_mode = True
                print("進入掃描模式")
            elif key == ord('n') and self.scanning_mode:
                if self.current_face < len(self.faces):
                    print(f"完成掃描面 {self.faces[self.current_face]}")
                    self.current_face += 1
                    if self.current_face >= len(self.faces):
                        self.scan_complete = True
                        print("所有面掃描完成!")
            elif key == ord('r'):
                if self.scan_complete:
                    print("開始求解...")
                    solution = self.solve_cube()
                    if solution:
                        print(f"解法: {solution}")
                    else:
                        print("無法求解，請檢查顏色掃描是否正確")
                else:
                    print("請先完成所有面的掃描")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    scanner = RubikCubeScanner()
    scanner.run()