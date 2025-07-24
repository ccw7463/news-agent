from abc import ABC, abstractmethod
from typing import Any, Optional
from logging import Logger


class ILoggerManager(ABC):
    """로거 매니저 인터페이스"""

    @abstractmethod
    def get_logger(self, name: str) -> Logger:
        """로거 인스턴스 반환"""
        pass

    @abstractmethod
    def setup_logger(
        self, name: str, log_level: str = "INFO", log_file: Optional[str] = None
    ) -> Logger:
        """로거 설정 및 반환"""
        pass

    @abstractmethod
    def close_logger(self, name: str) -> None:
        """로거 종료"""
        pass

    @abstractmethod
    def cleanup_old_log_files(self, log_dir: str, days: int = 7) -> None:
        """지정된 일수가 지난 로그 파일들 삭제"""
        pass

    @abstractmethod
    def get_log_file_path(self, name: str) -> Optional[str]:
        """로거의 로그 파일 경로 반환"""
        pass