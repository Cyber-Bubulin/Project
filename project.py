import cv2
import numpy as np
from statistics import mean


def calculate_real_size(pixel_size, distance_cm, focal_px):
    return (pixel_size * distance_cm) / focal_px


def get_corrected_size(rect, distance_cm, focal_px):
    # Размеры из minAreaRect
    (_, _), (w, h), angle = rect
    
    # Анализ угла наклона
    if angle < -45:
        w, h = h, w
        angle += 90
    
    # Определение ориентации
    if w > h:
        width_px = w
        height_px = h
    else:
        width_px = h
        height_px = w
    
    # Дополнительная проверка через соотношение сторон
    if width_px/height_px < 1.0:
        width_px, height_px = height_px, width_px
    
    # Расчёт реальных размеров
    width_cm = calculate_real_size(width_px, distance_cm, focal_px)
    height_cm = calculate_real_size(height_px, distance_cm, focal_px)
    
    return width_cm, height_cm
# Глобальные переменные для ROI
roi_selected = False
roi_box = None
tracking = False

def select_roi(event, x, y, flags, param):
    global roi_selected, roi_box, tracking
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_box = (x, y, 0, 0)
        roi_selected = True
    
    elif event == cv2.EVENT_MOUSEMOVE and roi_selected:
        roi_box = (roi_box[0], roi_box[1], x - roi_box[0], y - roi_box[1])
    
    elif event == cv2.EVENT_LBUTTONUP:
        roi_selected = False
        if roi_box[2] > 10 and roi_box[3] > 10:  # Минимальный размер ROI
            tracking = True
        else:
            roi_box = None

def main():
    global roi_box, tracking
    
    cap = cv2.VideoCapture(0)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Параметры камеры s
    focal_mm = 3.6  
    sensor_width_mm = 4.8  
    focal_px = (width * focal_mm) / sensor_width_mm  
    distance_cm = 50  # Начальное расстояние (можно менять +/-)
    
    cv2.namedWindow("Object Measurement")
    cv2.setMouseCallback("Object Measurement", select_roi)
    
    measurements = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Показываем инструкцию
        if not tracking:
            cv2.putText(frame, "Select an object", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Рисуем текущий ROI
        if roi_selected and roi_box:
            x, y, w, h = roi_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Обработка ROI после выделения
        if tracking and roi_box:
            x, y, w, h = roi_box
            roi_frame = frame[y:y+h, x:x+w].copy()
            
            # Детекция краёв
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Морфология для улучшения контура
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Поиск контуров
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:
                    # Получаем повёрнутый прямоугольник
                    rect = cv2.minAreaRect(largest_contour)
                    width_cm, height_cm = get_corrected_size(rect, distance_cm, focal_px)
                    # Подписываем размеры 
                    cv2.putText(frame, f"W: {width_cm:.1f} cm", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(frame, f"H: {height_cm:.1f} cm", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # Отображаем ROI для отладки
            cv2.imshow("ROI", roi_frame)
            cv2.imshow("Edges", edges)
        
        # Вывод расстояния
        cv2.putText(frame, f"Distance: {distance_cm} cm (+/- to adjust)", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Object Measurement", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('+'):
            distance_cm += 5
        elif key == ord('-'):
            distance_cm = max(5, distance_cm - 5)
        elif key == ord('r'):  # Сброс ROI
            tracking = False
            roi_box = None
            cv2.destroyWindow("ROI")
            cv2.destroyWindow("Edges")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()