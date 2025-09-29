#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
크기 분류 모듈
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
from datetime import datetime

class SizeClassifier:
    """딸기 크기 분류 클래스"""
    
    def __init__(self):
        # 기본 크기 기준 (mm³)
        self.size_criteria = {
            'small': {'min_volume': 0, 'max_volume': 15000, 'label': '소형', 'color': (0, 255, 0)},
            'medium': {'min_volume': 15000, 'max_volume': 30000, 'label': '중형', 'color': (0, 255, 255)},
            'large': {'min_volume': 30000, 'max_volume': 50000, 'label': '대형', 'color': (0, 165, 255)},
            'extra_large': {'min_volume': 50000, 'max_volume': float('inf'), 'label': '특대형', 'color': (0, 0, 255)}
        }
        
        # 품질 기준
        self.quality_criteria = {
            'excellent': {'min_confidence': 0.8, 'min_circularity': 0.7, 'label': '우수'},
            'good': {'min_confidence': 0.6, 'min_circularity': 0.5, 'label': '양호'},
            'fair': {'min_confidence': 0.4, 'min_circularity': 0.3, 'label': '보통'},
            'poor': {'min_confidence': 0.0, 'min_circularity': 0.0, 'label': '불량'}
        }
        
        # 통계 정보
        self.classification_stats = {
            'total_count': 0,
            'size_distribution': {size: 0 for size in self.size_criteria.keys()},
            'quality_distribution': {quality: 0 for quality in self.quality_criteria.keys()},
            'volume_statistics': {
                'mean': 0,
                'std': 0,
                'min': float('inf'),
                'max': 0
            }
        }
        
    def classify_by_volume(self, volume: float) -> str:
        """부피 기반 크기 분류"""
        for size_class, criteria in self.size_criteria.items():
            if criteria['min_volume'] <= volume < criteria['max_volume']:
                return size_class
        
        return 'medium'  # 기본값
    
    def classify_by_quality(self, result: Dict) -> str:
        """품질 기반 분류"""
        confidence = result.get('confidence', 0)
        top_detection = result.get('top_detection', {})
        side_detection = result.get('side_detection', {})
        
        # 평균 원형도 계산
        top_circularity = top_detection.get('circularity', 0)
        side_circularity = side_detection.get('circularity', 0)
        avg_circularity = (top_circularity + side_circularity) / 2
        
        for quality_class, criteria in self.quality_criteria.items():
            if (confidence >= criteria['min_confidence'] and 
                avg_circularity >= criteria['min_circularity']):
                return quality_class
        
        return 'poor'
    
    def calculate_grade(self, size_class: str, quality_class: str, result: Dict) -> str:
        """등급 계산"""
        # 크기 점수
        size_scores = {'small': 1, 'medium': 2, 'large': 3, 'extra_large': 4}
        size_score = size_scores.get(size_class, 2)
        
        # 품질 점수
        quality_scores = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
        quality_score = quality_scores.get(quality_class, 2)
        
        # 부피 신뢰도 점수
        confidence_score = result.get('confidence', 0.5) * 4
        
        # 종합 점수
        total_score = (size_score * 0.4 + quality_score * 0.4 + confidence_score * 0.2)
        
        # 등급 결정
        if total_score >= 3.5:
            return 'A+'
        elif total_score >= 3.0:
            return 'A'
        elif total_score >= 2.5:
            return 'B+'
        elif total_score >= 2.0:
            return 'B'
        elif total_score >= 1.5:
            return 'C+'
        elif total_score >= 1.0:
            return 'C'
        else:
            return 'D'
    
    def calculate_classification_confidence(self, result: Dict, size_class: str, 
                                          quality_class: str) -> float:
        """분류 신뢰도 계산"""
        # 부피 신뢰도
        volume_confidence = result.get('confidence', 0)
        
        # 크기 분류 신뢰도
        volume = result.get('best_volume', 0)
        size_criteria = self.size_criteria[size_class]
        
        class_center = (size_criteria['min_volume'] + size_criteria['max_volume']) / 2
        if size_criteria['max_volume'] == float('inf'):
            class_center = size_criteria['min_volume'] * 1.5
        
        volume_distance = abs(volume - class_center) / class_center
        size_confidence = max(0, 1 - volume_distance)
        
        # 품질 분류 신뢰도
        top_detection = result.get('top_detection', {})
        side_detection = result.get('side_detection', {})
        top_circularity = top_detection.get('circularity', 0)
        side_circularity = side_detection.get('circularity', 0)
        avg_circularity = (top_circularity + side_circularity) / 2
        quality_confidence = min(avg_circularity * 1.2, 1.0)
        
        # 종합 신뢰도
        total_confidence = (volume_confidence * 0.5 + 
                          size_confidence * 0.3 + 
                          quality_confidence * 0.2)
        
        return total_confidence
    
    def classify_sizes(self, volume_results: List[Dict]) -> List[Dict]:
        """크기 분류 메인 함수"""
        if not volume_results:
            return []
        
        classified_results = []
        
        for result in volume_results:
            # 크기 분류
            size_class = self.classify_by_volume(result['best_volume'])
            
            # 품질 분류
            quality_class = self.classify_by_quality(result)
            
            # 등급 계산
            grade = self.calculate_grade(size_class, quality_class, result)
            
            # 분류 결과 구성
            classified_result = result.copy()
            classified_result.update({
                'size_class': size_class,
                'size_label': self.size_criteria[size_class]['label'],
                'size_color': self.size_criteria[size_class]['color'],
                'quality_class': quality_class,
                'quality_label': self.quality_criteria[quality_class]['label'],
                'grade': grade,
                'classification_confidence': self.calculate_classification_confidence(result, size_class, quality_class)
            })
            
            classified_results.append(classified_result)
            
            # 통계 업데이트
            self.update_statistics(classified_result)
        
        return classified_results
    
    def update_statistics(self, classified_result: Dict):
        """통계 정보 업데이트"""
        self.classification_stats['total_count'] += 1
        
        # 크기 분포 업데이트
        size_class = classified_result['size_class']
        self.classification_stats['size_distribution'][size_class] += 1
        
        # 품질 분포 업데이트
        quality_class = classified_result['quality_class']
        self.classification_stats['quality_distribution'][quality_class] += 1
        
        # 부피 통계 업데이트
        volume = classified_result['best_volume']
        stats = self.classification_stats['volume_statistics']
        
        if volume < stats['min']:
            stats['min'] = volume
        if volume > stats['max']:
            stats['max'] = volume
    
    def get_classification_summary(self) -> Dict:
        """분류 결과 요약"""
        total_count = self.classification_stats['total_count']
        
        if total_count == 0:
            return {'total_count': 0}
        
        # 비율 계산
        size_ratios = {}
        for size, count in self.classification_stats['size_distribution'].items():
            size_ratios[size] = count / total_count
        
        quality_ratios = {}
        for quality, count in self.classification_stats['quality_distribution'].items():
            quality_ratios[quality] = count / total_count
        
        return {
            'total_count': total_count,
            'size_distribution': self.classification_stats['size_distribution'],
            'size_ratios': size_ratios,
            'quality_distribution': self.classification_stats['quality_distribution'],
            'quality_ratios': quality_ratios,
            'volume_statistics': self.classification_stats['volume_statistics']
        }
    
    def set_custom_criteria(self, size_criteria: Dict = None, quality_criteria: Dict = None):
        """사용자 정의 기준 설정"""
        if size_criteria:
            self.size_criteria.update(size_criteria)
            print("크기 기준이 업데이트되었습니다.")
        
        if quality_criteria:
            self.quality_criteria.update(quality_criteria)
            print("품질 기준이 업데이트되었습니다.")
    
    def reset_statistics(self):
        """통계 초기화"""
        self.classification_stats = {
            'total_count': 0,
            'size_distribution': {size: 0 for size in self.size_criteria.keys()},
            'quality_distribution': {quality: 0 for quality in self.quality_criteria.keys()},
            'volume_statistics': {
                'mean': 0,
                'std': 0,
                'min': float('inf'),
                'max': 0
            }
        }
        print("통계가 초기화되었습니다.")
    
    def export_classification_report(self, filepath: str, classified_results: List[Dict]):
        """분류 보고서 내보내기"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_classification_summary(),
            'detailed_results': []
        }
        
        for result in classified_results:
            detailed_result = {
                'id': result.get('id'),
                'size_class': result.get('size_class'),
                'size_label': result.get('size_label'),
                'quality_class': result.get('quality_class'),
                'quality_label': result.get('quality_label'),
                'grade': result.get('grade'),
                'volume': result.get('best_volume'),
                'confidence': result.get('confidence'),
                'classification_confidence': result.get('classification_confidence'),
                'dimensions': result.get('dimensions')
            }
            report['detailed_results'].append(detailed_result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"분류 보고서가 {filepath}에 저장되었습니다.")
