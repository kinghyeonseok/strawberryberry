#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
딸기 이중 카메라 부피 추정 및 크기 분류 시스템
메인 실행 파일
"""

import cv2
import numpy as np
import sys
import os
import yaml
from pathlib import Path
import time
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.calibration import CameraCalibrator
from src.measurement import VolumeEstimator
from src.classify import SizeClassifier
from src.utils import ImageProcessor, ResultVisualizer

class StrawberryDualCamSorter:
    """딸기 이중 카메라 분류 시스템"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self.load_config(config_path)
        
        # 컴포넌트 초기화
        self.calibrator = CameraCalibrator()
        self.volume_estimator = VolumeEstimator()
        self.size_classifier = SizeClassifier()
        self.image_processor = ImageProcessor()
        self.visualizer = ResultVisualizer()
        
        # 카메라 설정
        self.top_camera_id = self.config.get('cameras', {}).get('top_id', 0)
        self.side_camera_id = self.config.get('cameras', {}).get('side_id', 1)
        
        self.top_cap = None
        self.side_cap = None
        
        self.is_running = False
        self.results = []
        self.demo_mode = False
        
    def load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"설정 파일 로드 완료: {config_path}")
            return config
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return self.get_default_config()
        except Exception as e:
            print(f"설정 파일 로드 오류: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            'cameras': {
                'top_id': 0,
                'side_id': 1,
                'width': 1280,
                'height': 720,
                'fps': 30
            },
            'detection': {
                'min_area': 500,
                'max_area': 50000,
                'confidence_threshold': 0.5
            },
            'classification': {
                'size_criteria': {
                    'small': {'min_volume': 0, 'max_volume': 15000},
                    'medium': {'min_volume': 15000, 'max_volume': 30000},
                    'large': {'min_volume': 30000, 'max_volume': 50000},
                    'extra_large': {'min_volume': 50000, 'max_volume': float('inf')}
                }
            },
            'output': {
                'save_results': True,
                'save_images': True,
                'log_level': 'INFO'
            }
        }
    
    def initialize_cameras(self) -> bool:
        """카메라 초기화"""
        print("카메라를 초기화하는 중...")
        
        try:
            # Top 카메라 초기화
            self.top_cap = cv2.VideoCapture(self.top_camera_id)
            if not self.top_cap.isOpened():
                print(f"Top 카메라 (ID: {self.top_camera_id})를 열 수 없습니다.")
                return False
            
            # Side 카메라 초기화
            self.side_cap = cv2.VideoCapture(self.side_camera_id)
            if not self.side_cap.isOpened():
                print(f"Side 카메라 (ID: {self.side_camera_id})를 열 수 없습니다.")
                return False
            
            # 카메라 설정 적용
            camera_config = self.config.get('cameras', {})
            for cap, name in [(self.top_cap, "Top"), (self.side_cap, "Side")]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 1280))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 720))
                cap.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
            
            print("카메라 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"카메라 초기화 오류: {e}")
            return False
    
    def create_demo_images(self, frame_count: int = 0) -> tuple:
        """데모용 이미지 생성"""
        # Top 뷰 이미지 생성
        top_image = np.zeros((480, 640, 3), dtype=np.uint8)
        top_image[:] = (60, 60, 60)  # 어두운 배경
        
        # Side 뷰 이미지 생성
        side_image = np.zeros((480, 640, 3), dtype=np.uint8)
        side_image[:] = (60, 60, 60)  # 어두운 배경
        
        # 시간에 따른 애니메이션 효과
        t = frame_count * 0.1
        
        # 딸기 1
        x1 = int(200 + 50 * np.sin(t))
        y1 = int(200 + 30 * np.cos(t))
        radius1 = int(50 + 10 * np.sin(t * 2))
        cv2.circle(top_image, (x1, y1), radius1, (0, 0, 255), -1)
        cv2.circle(side_image, (x1, y1), int(radius1 * 0.8), (0, 0, 255), -1)
        
        # 딸기 2
        x2 = int(400 + 40 * np.cos(t * 1.5))
        y2 = int(300 + 20 * np.sin(t * 1.5))
        radius2 = int(45 + 8 * np.cos(t * 1.8))
        cv2.circle(top_image, (x2, y2), radius2, (0, 0, 200), -1)
        cv2.circle(side_image, (x2, y2), int(radius2 * 0.9), (0, 0, 200), -1)
        
        # 노이즈 추가
        noise = np.random.randint(0, 30, top_image.shape, dtype=np.uint8)
        top_image = cv2.add(top_image, noise)
        side_image = cv2.add(side_image, noise)
        
        return top_image, side_image
    
    def detect_strawberries(self, image: np.ndarray) -> list:
        """딸기 검출"""
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 딸기 색상 범위
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # 빨간색 마스크
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 형태학적 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        detection_config = self.config.get('detection', {})
        min_area = detection_config.get('min_area', 500)
        max_area = detection_config.get('max_area', 50000)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                # 원형도 계산
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                detection = {
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'center': center,
                    'area': area,
                    'circularity': circularity,
                    'confidence': min(circularity * 1.2, 1.0)
                }
                detections.append(detection)
        
        return detections
    
    def estimate_volume(self, top_detections: list, side_detections: list) -> list:
        """부피 추정"""
        volume_results = []
        
        # 간단한 매칭 (실제로는 더 정교한 매칭 필요)
        for i, (top_det, side_det) in enumerate(zip(top_detections, side_detections)):
            # 타원체 근사법으로 부피 계산
            top_radius = max(top_det['bbox'][2], top_det['bbox'][3]) / 2
            side_radius = max(side_det['bbox'][2], side_det['bbox'][3]) / 2
            
            # 픽셀을 mm로 변환 (실제로는 캘리브레이션 필요)
            pixel_to_mm = 0.1
            volume = (4/3) * np.pi * (top_radius * pixel_to_mm) * (top_radius * pixel_to_mm) * (side_radius * pixel_to_mm)
            
            result = {
                'id': i + 1,
                'top_detection': top_det,
                'side_detection': side_det,
                'volume': volume,
                'confidence': (top_det['confidence'] + side_det['confidence']) / 2
            }
            volume_results.append(result)
        
        return volume_results
    
    def classify_strawberries(self, volume_results: list) -> list:
        """딸기 분류"""
        classified_results = []
        size_criteria = self.config.get('classification', {}).get('size_criteria', {})
        
        for result in volume_results:
            volume = result['volume']
            
            # 크기 분류
            size_class = 'medium'  # 기본값
            for size, criteria in size_criteria.items():
                if criteria['min_volume'] <= volume < criteria['max_volume']:
                    size_class = size
                    break
            
            # 등급 계산
            confidence = result['confidence']
            if confidence > 0.8:
                grade = 'A+'
            elif confidence > 0.7:
                grade = 'A'
            elif confidence > 0.6:
                grade = 'B+'
            elif confidence > 0.5:
                grade = 'B'
            else:
                grade = 'C'
            
            classified_result = result.copy()
            classified_result.update({
                'size_class': size_class,
                'grade': grade,
                'size_label': size_class.capitalize()
            })
            
            classified_results.append(classified_result)
        
        return classified_results
    
    def process_frame(self, top_frame: np.ndarray, side_frame: np.ndarray) -> tuple:
        """프레임 처리"""
        try:
            # 딸기 검출
            top_detections = self.detect_strawberries(top_frame)
            side_detections = self.detect_strawberries(side_frame)
            
            if not top_detections or not side_detections:
                return None, top_frame, side_frame
            
            # 부피 추정
            volume_results = self.estimate_volume(top_detections, side_detections)
            
            # 크기 분류
            classified_results = self.classify_strawberries(volume_results)
            
            # 결과 시각화
            annotated_top = self.visualizer.draw_detections(top_frame, classified_results)
            annotated_side = self.visualizer.draw_detections(side_frame, classified_results)
            
            return classified_results, annotated_top, annotated_side
            
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            return None, top_frame, side_frame
    
    def display_results(self, top_frame: np.ndarray, side_frame: np.ndarray, results: list = None):
        """결과 화면 표시"""
        # 화면 크기 조정
        display_top = cv2.resize(top_frame, (640, 480))
        display_side = cv2.resize(side_frame, (640, 480))
        
        # 화면 합치기
        combined = np.hstack([display_top, display_side])
        
        # 제목 추가
        mode_text = " (Demo Mode)" if self.demo_mode else ""
        cv2.putText(combined, f"Top Camera{mode_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, f"Side Camera{mode_text}", (650, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 조작 안내
        cv2.putText(combined, "Press 'q' to quit, 's' to save, 'c' to calibrate", 
                   (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 결과 정보 표시
        if results:
            info_text = f"Detected: {len(results)} strawberries"
            cv2.putText(combined, info_text, (10, 480), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Strawberry Dual Camera Sorter", combined)
    
    def save_results(self, results: list):
        """결과 저장"""
        if not results:
            print("저장할 결과가 없습니다.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/analysis_{timestamp}.json"
        
        # 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        
        # 결과 저장
        save_data = {
            'timestamp': timestamp,
            'total_count': len(results),
            'results': results
        }
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"결과가 {filename}에 저장되었습니다.")
    
    def run(self):
        """메인 실행 루프"""
        print("=" * 60)
        print("딸기 이중 카메라 분류 시스템")
        print("=" * 60)
        
        # 카메라 초기화
        if not self.initialize_cameras():
            print("카메라 초기화 실패 - 데모 모드로 전환합니다")
            self.demo_mode = True
        else:
            self.demo_mode = False
        
        self.is_running = True
        print("시스템 시작! 'q'를 눌러 종료하세요.")
        
        frame_count = 0
        
        while self.is_running:
            if self.demo_mode:
                # 데모 모드: 가상 이미지 생성
                top_frame, side_frame = self.create_demo_images(frame_count)
            else:
                # 실제 카메라 모드
                ret_top, top_frame = self.top_cap.read()
                ret_side, side_frame = self.side_cap.read()
                
                if not ret_top or not ret_side:
                    print("프레임 캡처 실패")
                    continue
            
            # 프레임 처리
            results, annotated_top, annotated_side = self.process_frame(top_frame, side_frame)
            
            if results:
                self.results.extend(results)
                print(f"검출된 딸기: {len(results)}개")
                for result in results:
                    volume = result.get('volume', 0)
                    size_label = result.get('size_label', 'Unknown')
                    grade = result.get('grade', 'N/A')
                    print(f"  딸기 {result['id']}: {size_label} ({grade}) - {volume:.2f}mm³")
            
            # 결과 표시
            self.display_results(annotated_top, annotated_side, results)
            
            # 키 입력 확인
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_results(self.results)
            elif key == ord('c'):
                self.show_calibration_info()
            
            frame_count += 1
        
        self.cleanup()
    
    def show_calibration_info(self):
        """캘리브레이션 정보 표시"""
        print("\n캘리브레이션 정보:")
        print("  픽셀/mm 비율: 0.1 (기본값)")
        print("  실제 사용 시에는 체스보드 패턴으로 캘리브레이션을 수행하세요.")
    
    def cleanup(self):
        """리소스 정리"""
        print("시스템을 종료하는 중...")
        
        if self.top_cap:
            self.top_cap.release()
        if self.side_cap:
            self.side_cap.release()
        
        cv2.destroyAllWindows()
        
        # 최종 통계 출력
        if self.results:
            print(f"\n최종 통계:")
            print(f"  총 처리된 딸기: {len(self.results)}개")
            
            # 크기 분포
            size_dist = {}
            for result in self.results:
                size = result.get('size_class', 'unknown')
                size_dist[size] = size_dist.get(size, 0) + 1
            
            print(f"  크기 분포: {size_dist}")
        
        print("시스템 종료 완료!")

def main():
    """메인 함수"""
    try:
        sorter = StrawberryDualCamSorter()
        sorter.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
