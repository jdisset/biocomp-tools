# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import argparse
from pathlib import Path
import concurrent.futures
import subprocess


def make_video(input_file_pattern, output_file, fps=30, crf=17, vcodec='libx264'):
    cmd = f'ffmpeg -y -r {fps} -i "{input_file_pattern}" -crf {crf} -vcodec {vcodec} -vf "scale=iw:ih,format=yuv420p,crop=trunc(iw/2)*2:trunc(ih/2)*2" "{output_file}"'
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'processed {output_file}')


def process_frame_dir(params):
    make_video(**params)


def main():

    parser = argparse.ArgumentParser(description='Create video from  ,frame images')
    parser.add_argument(
        '--root_dir',
        type=str,
        default='./output_plot',
        help='Root directory to search for frame directories',
    )
    parser.add_argument(
        '--dir_pattern', type=str, default='*3Dframes', help='Pattern to match frame directories'
    )
    parser.add_argument(
        '--fps', type=int, default=15, help='Frames per second for the output video'
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    dir_pattern = args.dir_pattern
    fps = args.fps

    params = [
        dict(
            input_file_pattern=f'{frame_dir}/frame_%d.png', output_file=f'{frame_dir}.mp4', fps=fps
        )
        for frame_dir in root_dir.glob(dir_pattern)
        # [Path(p).resolve() for p in ]
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_frame_dir, params)


if __name__ == '__main__':
    main()
