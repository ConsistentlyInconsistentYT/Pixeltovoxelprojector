"""
Pipeline Package - Processing pipeline and coordination

This package contains the main processing pipeline and coordinator
for orchestrating the entire motion tracking system.
"""

from .processing_pipeline import ProcessingPipeline, PipelineConfig, FrameData, ProcessingResult
from .pipeline_coordinator import PipelineCoordinator, CoordinatorConfig, ComponentState, ComponentStatus

__all__ = [
    'ProcessingPipeline',
    'PipelineConfig',
    'FrameData',
    'ProcessingResult',
    'PipelineCoordinator',
    'CoordinatorConfig',
    'ComponentState',
    'ComponentStatus'
]
