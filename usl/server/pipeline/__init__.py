from .server_stage import ServerPipelineStage
from .usl_gpipe import ServerScheduleGPipe
from .usl_1f1b import ServerSchedule1F1B
from .base_schedule import ServerPipelineScheduleSingle

__all__ = [
    "ServerPipelineStage",
    "ServerScheduleGPipe",
    "ServerSchedule1F1B",
    "ServerPipelineScheduleSingle",
    "get_schedule_class",
]


def get_schedule_class(schedule_name: str):
    """
    Maps a schedule name (case insensitive) to its corresponding class object.

    Args:
        schedule_name (str): The name of the schedule.
    """
    schedule_map = {
        "1F1B": ServerSchedule1F1B,
        "GPipe": ServerScheduleGPipe,
        "PipelineScheduleSingle": ServerPipelineScheduleSingle,
    }
    lowercase_keys = {k.lower(): k for k in schedule_map.keys()}
    lowercase_schedule_name = schedule_name.lower()
    if lowercase_schedule_name not in lowercase_keys:
        raise ValueError(f"Unknown schedule name '{schedule_name}'. The valid options are {list(schedule_map.keys())}")
    return schedule_map[lowercase_keys[lowercase_schedule_name]]
