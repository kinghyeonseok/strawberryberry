#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카메라 캘리브레이션 모듈
"""

import cv2
import numpy as np
import yaml
import os
from typing import Tuple, List, Dict, Optional

class CameraCalibrator:
    """카메라 캘리브레이션 클래스"""
    
    def __init__(self):
        self.chessboard_size = (9, 6)
        self.square_size = 25.0  # mm
        
        # 캘리브레이션 결과
        self.camera_matrix_top = None
        self.dist_coeffs_top = None
        self.camera_matrix_side = None
        self.dist_coeffs_side = None
        
        # 스테레오 캘리브레이션 결과
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        
    def find_chessboard_corners(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """체스보드 코너 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(
            gray, self.chessboard_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    def calibrate_cameras(self, image_dir: str) -> Dict:
        """카메라 캘리브레이션 수행"""
        # 3D 점 준비
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        objpoints = []
        imgpoints_top = []
        imgpoints_side = []
        
        # 이미지 파일 목록
        top_files = sorted([f for f in os.listdir(image_dir) if f.startswith('top_')])
        side_files = sorted([f for f in os.listdir(image_dir) if f.startswith('side_')])
        
        for top_file, side_file in zip(top_files, side_files):
            top_path = os.path.join(image_dir, top_file)
            side_path = os.path.join(image_dir, side_file)
            
            top_img = cv2.imread(top_path)
            side_img = cv2.imread(side_path)
            
            if top_img is None or side_img is None:
                continue
            
            ret_top, corners_top = self.find_chessboard_corners(top_img)
            ret_side, corners_side = self.find_chessboard_corners(side_img)
            
            if ret_top and ret_side:
                objpoints.append(objp)
                imgpoints_top.append(corners_top)
                imgpoints_side.append(corners_side)
        
        if len(objpoints) < 10:
            raise ValueError("캘리브레이션에 충분한 이미지가 없습니다 (최소 10장 필요)")
        
        # 이미지 크기
        image_shape = (top_img.shape[1], top_img.shape[0])
        
        # 단일 카메라 캘리브레이션
        ret, self.camera_matrix_top, self.dist_coeffs_top, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_top, image_shape, None, None
        )
        
        ret, self.camera_matrix_side, self.dist_coeffs_side, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_side, image_shape, None, None
        )
        
        # 스테레오 캘리브레이션
        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints, imgpoints_top, imgpoints_side,
            self.camera_matrix_top, self.dist_coeffs_top,
            self.camera_matrix_side, self.dist_coeffs_side,
            image_shape
        )
        
        return {
            'camera_matrix_top': self.camera_matrix_top.tolist(),
            'dist_coeffs_top': self.dist_coeffs_top.tolist(),
            'camera_matrix_side': self.camera_matrix_side.tolist(),
            'dist_coeffs_side': self.dist_coeffs_side.tolist(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'E': self.E.tolist(),
            'F': self.F.tolist()
        }
    
    def save_calibration(self, filepath: str, calibration_data: Dict):
        """캘리브레이션 결과 저장"""
        with open(filepath, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)
        print(f"캘리브레이션 결과가 {filepath}에 저장되었습니다")
    
    def load_calibration(self, filepath: str):
        """캘리브레이션 결과 로드"""
        with open(filepath, 'r') as f:
            calibration_data = yaml.safe_load(f)
        
        self.camera_matrix_top = np.array(calibration_data['camera_matrix_top'])
        self.dist_coeffs_top = np.array(calibration_data['dist_coeffs_top'])
        self.camera_matrix_side = np.array(calibration_data['camera_matrix_side'])
        self.dist_coeffs_side = np.array(calibration_data['dist_coeffs_side'])
        self.R = np.array(calibration_data['R'])
        self.T = np.array(calibration_data['T'])
        self.E = np.array(calibration_data['E'])
        self.F = np.array(calibration_data['F'])
        
        print(f"캘리브레이션 결과를 {filepath}에서 로드했습니다")
    
    def undistort_image(self, image: np.ndarray, camera_type: str) -> np.ndarray:
        """이미지 보정"""
        if camera_type.lower() == "top" and self.camera_matrix_top is not None:
            return cv2.undistort(image, self.camera_matrix_top, self.dist_coeffs_top)
        elif camera_type.lower() == "side" and self.camera_matrix_side is not None:
            return cv2.undistort(image, self.camera_matrix_side, self.dist_coeffs_side)
        
        return image
