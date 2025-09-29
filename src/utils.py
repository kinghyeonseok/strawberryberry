#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
유틸리티 모듈
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

class ImageProcessor:
    """이미지 처리 클래스"""
    
    def __init__(self):
        self.default_blur_kernel = (5, 5)
        self.default_morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def enhance_image(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """이미지 향상"""
        if method == 'clahe':
            return self._clahe_enhancement(image)
        elif method == 'histogram_equalization':
            return self._histogram_equalization(image)
        elif method == 'gamma_correction':
            return self._gamma_correction(image)
        else:
            return image
    
    def _clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """히스토그램 균등화"""
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """감마 보정"""
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, lookup_table)
    
    def detect_edges(self, image: np.ndarray, method: str = 'canny') -> np.ndarray:
        """엣지 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == 'canny':
            return cv2.Canny(gray, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        elif method == 'laplacian':
            return cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)
        else:
            return gray
    
    def apply_morphological_operations(self, image: np.ndarray, 
                                     operations: List[str] = ['opening', 'closing']) -> np.ndarray:
        """형태학적 연산 적용"""
        result = image.copy()
        
        for operation in operations:
            if operation == 'opening':
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.default_morph_kernel)
            elif operation == 'closing':
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, self.default_morph_kernel)
            elif operation == 'erosion':
                result = cv2.erode(result, self.default_morph_kernel, iterations=1)
            elif operation == 'dilation':
                result = cv2.dilate(result, self.default_morph_kernel, iterations=1)
        
        return result
    
    def find_contours(self, image: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
        """윤곽선 검출"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 최소 면적 필터링
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        return filtered_contours
    
    def calculate_contour_features(self, contour: np.ndarray) -> dict:
        """윤곽선 특징 계산"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 경계 상자
        x, y, w, h = cv2.boundingRect(contour)
        
        # 최소 외접 원
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        
        # 타원 피팅
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
        else:
            ellipse = None
        
        # 원형도
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 종횡비
        aspect_ratio = w / h if h > 0 else 0
        
        # 볼록 껍질
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'bounding_rect': (x, y, w, h),
            'center': (int(center_x), int(center_y)),
            'radius': radius,
            'ellipse': ellipse,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        }

class ResultVisualizer:
    """결과 시각화 클래스"""
    
    def __init__(self):
        # 색상 팔레트
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        
        # 크기별 색상
        self.size_colors = {
            'small': (0, 255, 0),      # 녹색
            'medium': (0, 255, 255),   # 노란색
            'large': (0, 165, 255),    # 주황색
            'extra_large': (0, 0, 255) # 빨간색
        }
        
        # 품질별 색상
        self.quality_colors = {
            'excellent': (0, 255, 0),  # 녹색
            'good': (0, 255, 255),     # 노란색
            'fair': (0, 165, 255),     # 주황색
            'poor': (0, 0, 255)        # 빨간색
        }
        
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_confidence: bool = True, show_size: bool = True) -> np.ndarray:
        """검출 결과 시각화"""
        result_image = image.copy()
        
        for detection in detections:
            # 기본 정보 추출
            bbox = detection.get('bbox', (0, 0, 0, 0))
            center = detection.get('center', (0, 0))
            confidence = detection.get('confidence', 0)
            size_class = detection.get('size_class', 'medium')
            quality_class = detection.get('quality_class', 'fair')
            
            # 색상 결정
            if show_size and size_class in self.size_colors:
                color = self.size_colors[size_class]
            elif quality_class in self.quality_colors:
                color = self.quality_colors[quality_class]
            else:
                color = self.colors['green']
            
            # 경계 상자 그리기
            x, y, w, h = bbox
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # 중심점 그리기
            cv2.circle(result_image, center, 3, color, -1)
            
            # 반지름이 있으면 원 그리기
            if 'radius' in detection:
                radius = detection['radius']
                cv2.circle(result_image, center, radius, color, 1)
            
            # 라벨 생성
            label_parts = [f"#{detection.get('id', 1)}"]
            
            if show_size and 'size_label' in detection:
                label_parts.append(detection['size_label'])
            
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # 라벨 배경
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # 라벨 텍스트
            cv2.putText(result_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        return result_image
    
    def draw_volume_info(self, image: np.ndarray, volume_results: List[Dict]) -> np.ndarray:
        """부피 정보 시각화"""
        result_image = image.copy()
        
        for result in volume_results:
            bbox = result.get('bbox', (0, 0, 0, 0))
            volume = result.get('best_volume', 0)
            confidence = result.get('confidence', 0)
            
            # 부피 정보 텍스트
            volume_text = f"Vol: {volume:.0f}mm³"
            conf_text = f"Conf: {confidence:.2f}"
            
            x, y, w, h = bbox
            
            # 텍스트 배경
            text_y = y + h + 20
            cv2.rectangle(result_image, (x, text_y - 15), (x + 150, text_y + 5), 
                         self.colors['black'], -1)
            
            # 텍스트 그리기
            cv2.putText(result_image, volume_text, (x + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
            cv2.putText(result_image, conf_text, (x + 5, text_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        
        return result_image
    
    def draw_classification_info(self, image: np.ndarray, classified_results: List[Dict]) -> np.ndarray:
        """분류 정보 시각화"""
        result_image = image.copy()
        
        for result in classified_results:
            bbox = result.get('bbox', (0, 0, 0, 0))
            size_label = result.get('size_label', 'Unknown')
            quality_label = result.get('quality_label', 'Unknown')
            grade = result.get('grade', 'N/A')
            
            x, y, w, h = bbox
            
            # 분류 정보 텍스트
            info_text = f"{size_label} | {quality_label} | {grade}"
            
            # 텍스트 배경
            text_y = y - 10
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_image, (x, text_y - text_size[1] - 5), 
                         (x + text_size[0] + 10, text_y + 5), self.colors['black'], -1)
            
            # 텍스트 그리기
            cv2.putText(result_image, info_text, (x + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        
        return result_image
    
    def create_summary_overlay(self, image: np.ndarray, summary: Dict) -> np.ndarray:
        """요약 정보 오버레이"""
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        # 오버레이 배경
        overlay = np.zeros((200, 300, 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # 검은색 배경
        
        # 요약 정보 텍스트
        y_offset = 30
        line_height = 25
        
        # 제목
        cv2.putText(overlay, "Analysis Summary", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['white'], 2)
        y_offset += line_height * 2
        
        # 총 개수
        total_count = summary.get('total_count', 0)
        cv2.putText(overlay, f"Total: {total_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 1)
        y_offset += line_height
        
        # 크기 분포
        size_dist = summary.get('size_distribution', {})
        for size, count in size_dist.items():
            if count > 0:
                size_label = size.capitalize()
                cv2.putText(overlay, f"{size_label}: {count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.size_colors.get(size, self.colors['white']), 1)
                y_offset += line_height
        
        # 오버레이를 이미지에 합성
        result_image[10:210, w-310:w-10] = overlay
        
        return result_image
    
    def save_visualization_results(self, image: np.ndarray, filepath: str, 
                                 metadata: Optional[Dict] = None):
        """시각화 결과 저장"""
        # 이미지 저장
        cv2.imwrite(filepath, image)
        
        # 메타데이터 저장
        if metadata:
            metadata_path = filepath.replace('.jpg', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"시각화 결과가 {filepath}에 저장되었습니다.")
