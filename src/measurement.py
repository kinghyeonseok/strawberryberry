#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
부피 측정 모듈
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional

class VolumeEstimator:
    """부피 추정 클래스"""
    
    def __init__(self):
        self.pixel_to_mm_ratio = 0.1  # 기본값, 캘리브레이션으로 업데이트
        self.camera_height = 500  # mm
        self.camera_distance = 300  # mm
        
    def estimate_volume_ellipsoid(self, top_detection: Dict, side_detection: Dict) -> float:
        """타원체 근사법으로 부피 추정"""
        # Top 뷰에서의 반지름
        top_radius_x = top_detection['bbox'][2] / 2 * self.pixel_to_mm_ratio
        top_radius_y = top_detection['bbox'][3] / 2 * self.pixel_to_mm_ratio
        
        # Side 뷰에서의 반지름 (z 방향)
        side_radius_z = side_detection['bbox'][3] / 2 * self.pixel_to_mm_ratio
        
        # 타원체 부피 공식: V = (4/3) * π * a * b * c
        volume = (4/3) * math.pi * top_radius_x * top_radius_y * side_radius_z
        
        return volume
    
    def estimate_volume_sphere(self, top_detection: Dict, side_detection: Dict) -> float:
        """구형 근사법으로 부피 추정"""
        # 평균 반지름 계산
        top_diameter = max(top_detection['bbox'][2], top_detection['bbox'][3])
        side_diameter = max(side_detection['bbox'][2], side_detection['bbox'][3])
        
        avg_diameter = (top_diameter + side_diameter) / 2
        radius = (avg_diameter / 2) * self.pixel_to_mm_ratio
        
        # 구 부피 공식: V = (4/3) * π * r³
        volume = (4/3) * math.pi * radius**3
        
        return volume
    
    def estimate_volume_contour(self, top_frame: np.ndarray, side_frame: np.ndarray, 
                              top_detection: Dict, side_detection: Dict) -> float:
        """윤곽선 기반 부피 추정"""
        # Top 뷰 윤곽선 면적
        top_contour = top_detection.get('contour')
        if top_contour is not None:
            top_area = cv2.contourArea(top_contour) * (self.pixel_to_mm_ratio ** 2)
        else:
            top_area = top_detection['area'] * (self.pixel_to_mm_ratio ** 2)
        
        # Side 뷰 높이
        side_height = side_detection['bbox'][3] * self.pixel_to_mm_ratio
        
        # 원기둥 근사
        radius = math.sqrt(top_area / math.pi)
        volume = math.pi * radius**2 * side_height
        
        # 타원체 보정 계수
        volume *= 0.75
        
        return volume
    
    def estimate_volume_depth(self, top_detection: Dict, side_detection: Dict) -> float:
        """깊이 추정법으로 부피 추정"""
        # Top 뷰에서의 크기
        top_diameter = max(top_detection['bbox'][2], top_detection['bbox'][3]) * self.pixel_to_mm_ratio
        
        # Side 뷰에서의 높이
        side_height = side_detection['bbox'][3] * self.pixel_to_mm_ratio
        
        # 깊이 추정
        depth = self.estimate_depth_from_side_view(side_detection)
        
        # 구형 근사
        radius = top_diameter / 2
        volume = (4/3) * math.pi * radius**3
        
        # 깊이 보정
        depth_factor = min(depth / side_height, 2.0)
        volume *= depth_factor
        
        return volume
    
    def estimate_depth_from_side_view(self, side_detection: Dict) -> float:
        """Side 뷰에서 깊이 추정"""
        side_height_pixels = side_detection['bbox'][3]
        side_height_mm = side_height_pixels * self.pixel_to_mm_ratio
        
        # 카메라 각도와 높이를 고려한 깊이 계산
        depth = side_height_mm * 1.2  # 경험적 보정 계수
        
        return depth
    
    def estimate_volume(self, top_frame: np.ndarray, side_frame: np.ndarray,
                       top_detections: List[Dict], side_detections: List[Dict]) -> List[Dict]:
        """부피 추정 메인 함수"""
        if not top_detections or not side_detections:
            return []
        
        volume_results = []
        
        # 간단한 매칭 (실제로는 더 정교한 매칭 필요)
        for i, (top_det, side_det) in enumerate(zip(top_detections, side_detections)):
            try:
                # 여러 방법으로 부피 추정
                volume_ellipsoid = self.estimate_volume_ellipsoid(top_det, side_det)
                volume_sphere = self.estimate_volume_sphere(top_det, side_det)
                volume_contour = self.estimate_volume_contour(top_frame, side_frame, top_det, side_det)
                volume_depth = self.estimate_volume_depth(top_det, side_det)
                
                # 최적 추정값 선택 (중간값)
                volumes = [volume_ellipsoid, volume_sphere, volume_contour, volume_depth]
                volumes = [v for v in volumes if v > 0]  # 유효한 값만
                
                if volumes:
                    volumes.sort()
                    best_volume = volumes[len(volumes)//2]  # 중간값
                else:
                    best_volume = 0
                
                # 결과 구성
                result = {
                    'id': i + 1,
                    'top_detection': top_det,
                    'side_detection': side_det,
                    'volume_estimates': {
                        'ellipsoid': volume_ellipsoid,
                        'sphere': volume_sphere,
                        'contour': volume_contour,
                        'depth': volume_depth
                    },
                    'best_volume': best_volume,
                    'confidence': self.calculate_volume_confidence(volumes, best_volume),
                    'dimensions': self.estimate_dimensions(top_det, side_det)
                }
                
                volume_results.append(result)
                
            except Exception as e:
                print(f"부피 추정 중 오류: {e}")
                continue
        
        return volume_results
    
    def calculate_volume_confidence(self, volumes: List[float], best_volume: float) -> float:
        """부피 추정 신뢰도 계산"""
        if len(volumes) < 2:
            return 0.5
        
        # 추정값들의 일관성 계산
        variance = np.var(volumes)
        mean_volume = np.mean(volumes)
        
        # 변동계수 (CV) 계산
        if mean_volume > 0:
            cv = math.sqrt(variance) / mean_volume
            confidence = max(0, 1 - cv)  # 변동이 적을수록 높은 신뢰도
        else:
            confidence = 0.0
        
        return confidence
    
    def estimate_dimensions(self, top_detection: Dict, side_detection: Dict) -> Dict:
        """딸기 치수 추정"""
        # Top 뷰에서의 치수
        top_width = top_detection['bbox'][2] * self.pixel_to_mm_ratio
        top_height = top_detection['bbox'][3] * self.pixel_to_mm_ratio
        
        # Side 뷰에서의 치수
        side_width = side_detection['bbox'][2] * self.pixel_to_mm_ratio
        side_depth = side_detection['bbox'][3] * self.pixel_to_mm_ratio
        
        # 평균 치수 계산
        width = (top_width + side_width) / 2
        height = top_height
        depth = side_depth
        
        return {
            'width': width,
            'height': height,
            'depth': depth,
            'diameter': (width + height) / 2,
            'aspect_ratio': width / height if height > 0 else 1.0
        }
    
    def set_calibration_parameters(self, pixel_to_mm_ratio: float, camera_height: float, 
                                 camera_distance: float):
        """캘리브레이션 파라미터 설정"""
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.camera_height = camera_height
        self.camera_distance = camera_distance
        
        print(f"캘리브레이션 파라미터 설정: 픽셀/mm={pixel_to_mm_ratio}, 카메라 높이={camera_height}mm, 카메라 거리={camera_distance}mm")
