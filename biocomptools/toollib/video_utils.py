import subprocess
from pathlib import Path
from typing import Optional, Union
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


DEFAULT_FRAMERATE = 4.0
DEFAULT_CRF = 20
DEFAULT_MIN_PLOTS = 2


logger = get_logger(__name__)


def create_video_from_plots(
    plot_dir: Union[str, Path],
    output_path: Union[str, Path],
    plot_pattern: str = "*.png",
    framerate: float = DEFAULT_FRAMERATE,
    crf: int = DEFAULT_CRF,
    min_plots: int = DEFAULT_MIN_PLOTS,
) -> bool:
    """
    Create an MP4 video from a directory of plot images using ffmpeg.

    Args:
        plot_dir: Directory containing plot images
        output_path: Path for output video file
        plot_pattern: Glob pattern for plot files (default: "*.png")
        framerate: Video framerate in FPS (default: DEFAULT_FRAMERATE)
        crf: Constant Rate Factor for video quality, 0-51 where lower=better (default: DEFAULT_CRF)
        min_plots: Minimum number of plots required to create video (default: DEFAULT_MIN_PLOTS)

    Returns:
        True if video was created successfully, False otherwise
    """
    plot_dir = Path(plot_dir)
    output_path = Path(output_path)

    if not plot_dir.exists():
        logger.debug(f"Plot directory does not exist: {plot_dir}")
        return False

    # Find plot files
    plot_files = sorted(plot_dir.glob(plot_pattern))
    if len(plot_files) < min_plots:
        logger.debug(
            f"Insufficient plots in {plot_dir} - found {len(plot_files)}, need at least {min_plots}"
        )
        return False

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # ffmpeg command to create video from images
        cmd = [
            'ffmpeg',
            '-y',  # -y to overwrite existing files
            '-framerate',
            str(framerate),
            '-pattern_type',
            'glob',
            '-i',
            str(plot_dir / plot_pattern),
            '-c:v',
            'libx264',
            '-crf',
            str(crf),
            '-pix_fmt',
            'yuv420p',
            '-vf',
            'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # ensure even dimensions
            str(output_path),
        ]

        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=plot_dir)

        if result.returncode == 0:
            logger.info(f"Created video: {output_path}")
            return True
        else:
            logger.warning(f"Failed to create video {output_path}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error creating video {output_path}: {e}")
        return False


def create_videos_from_subdirs(
    base_dir: Union[str, Path],
    subdir_pattern: str = "replicate*",
    plot_pattern: str = "*.png",
    video_name: str = "video.mp4",
    framerate: float = DEFAULT_FRAMERATE,
    crf: int = DEFAULT_CRF,
    min_plots: int = DEFAULT_MIN_PLOTS,
) -> int:
    """
    Create videos from plots in multiple subdirectories.

    Args:
        base_dir: Base directory containing subdirectories with plots
        subdir_pattern: Glob pattern for subdirectories (default: "replicate*")
        plot_pattern: Glob pattern for plot files (default: "*.png")
        video_name: Name for output video files (default: "video.mp4")
        framerate: Video framerate in FPS (default: DEFAULT_FRAMERATE)
        crf: Constant Rate Factor for video quality (default: DEFAULT_CRF)
        min_plots: Minimum number of plots required to create video (default: DEFAULT_MIN_PLOTS)

    Returns:
        Number of videos successfully created
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        logger.debug(f"Base directory does not exist: {base_dir}")
        return 0

    # Find subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.match(subdir_pattern)]

    if not subdirs:
        logger.debug(f"No subdirectories matching '{subdir_pattern}' found in {base_dir}")
        return 0

    videos_created = 0
    for subdir in subdirs:
        video_path = subdir / video_name
        if create_video_from_plots(
            plot_dir=subdir,
            output_path=video_path,
            plot_pattern=plot_pattern,
            framerate=framerate,
            crf=crf,
            min_plots=min_plots,
        ):
            videos_created += 1

    return videos_created
