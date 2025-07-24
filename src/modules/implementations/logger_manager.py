import os
import logging
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict
import pytz
from src.modules.interfaces.logger_manager import ILoggerManager


class KSTFormatter(logging.Formatter):
    """한국시간대를 사용하는 로그 포맷터"""

    KST = pytz.timezone("Asia/Seoul")

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=self.KST)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")


class LoggerManager(ILoggerManager):
    """
    Des:
        로거 매니저 구현체
        - 크롤러별 로거 설정 및 관리 기능 제공
        - 콘솔 및 파일 로깅 지원
        - 오래된 로그 파일 정리 기능 제공
    """

    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, list] = {}
        self.log_file_paths: Dict[str, str] = {}  # 로거별 로그 파일 경로 저장
        self.korea_tz = pytz.timezone("Asia/Seoul")

    def get_logger(self, name: str) -> logging.Logger:
        """로거 인스턴스 반환"""
        if name in self.loggers:
            return self.loggers[name]
        else:
            return self.setup_logger(name)

    def setup_logger(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = False,
    ) -> logging.Logger:
        """로거 설정 및 반환"""
        # 이미 존재하는 로거인지 확인
        if name in self.loggers:
            return self.loggers[name]

        # 로거 생성
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # 기존 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 포맷터 설정 (한국시간대 사용)
        formatter = KSTFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 콘솔 핸들러 추가 (console_output이 True인 경우에만)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 파일 핸들러 추가 (지정된 경우)
        if log_file:
            # 로그 디렉토리 생성
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 로거 저장
        self.loggers[name] = logger
        self.handlers[name] = logger.handlers
        
        # 로그 파일 경로 저장 (None이 아닌 경우에만)
        if log_file:
            self.log_file_paths[name] = log_file

        return logger

    def close_logger(self, name: str) -> None:
        """로거 종료"""
        if name in self.loggers:
            logger = self.loggers[name]

            # 모든 핸들러 제거
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            # 로거 제거
            del self.loggers[name]
            if name in self.handlers:
                del self.handlers[name]
            if name in self.log_file_paths:
                del self.log_file_paths[name]

    def get_log_file_path(self, name: str) -> Optional[str]:
        """지정된 로거의 로그 파일 경로를 반환합니다."""
        return self.log_file_paths.get(name)

    def cleanup_old_log_files(self, log_dir: str, days: int = 7) -> None:
        """
        지정된 일수가 지난 로그 파일들 삭제
        
        Args:
            log_dir: 로그 디렉토리 경로
            days: 삭제 기준 일수 (기본값: 7일)
        """
        try:
            # log_dir 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                return
            
            # 현재 시간 (한국 시간)
            current_time = datetime.now(self.korea_tz)
            cutoff_date = current_time - timedelta(days=days)
            
            # log_dir 폴더의 모든 .log 파일 검색
            log_pattern = os.path.join(log_dir, "*.log")
            log_files = glob.glob(log_pattern)
            
            deleted_count = 0
            total_size_freed = 0
            
            for log_file in log_files:
                try:
                    # 파일의 수정 시간 확인
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(log_file), self.korea_tz)
                    
                    # 지정된 일수가 지난 파일인지 확인
                    if file_mtime < cutoff_date:
                        # 파일 크기 확인 (삭제 전)
                        file_size = os.path.getsize(log_file)
                        
                        # 파일 삭제
                        os.remove(log_file)
                        deleted_count += 1
                        total_size_freed += file_size
                        
                        print(f"오래된 로그 파일 삭제: {os.path.basename(log_file)} "
                              f"(수정일: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                        
                except (OSError, IOError) as e:
                    print(f"⚠️ 로그 파일 삭제 중 오류 발생: {os.path.basename(log_file)} - {str(e)}")
                    continue
            
            if deleted_count > 0:
                size_mb = total_size_freed / (1024 * 1024)
                print(f"✅ 로그 파일 정리 완료: {deleted_count}개 파일 삭제, {size_mb:.2f}MB 공간 확보")
            else:
                print("삭제할 오래된 로그 파일이 없습니다")
                
        except Exception as e:
            print(f"❌ 로그 파일 정리 중 오류 발생: {str(e)}")