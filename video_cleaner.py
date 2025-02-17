import argparse
import shutil
import os
from typing import Tuple, Optional

import cv2
import numpy as np

from src.FrameFilter import FrameFilter
from src.FrameOrderer import FrameOrderer
from src.encoders.OrbEncoder import OrbEncoder
from src.encoders.VitEncoder import VitEncoder


def create_arg_parser():
    parser = argparse.ArgumentParser(
        prog='video_cleaner',
        description='This program cleans and order a video',
    )
    parser.add_argument('filename', type=str, help="Path to the file to be read")
    parser.add_argument("-e", "--encoder", help="Which encoder to use (vit, orb)", default="vit")
    parser.add_argument("-o", "--output", help="path of the output video", default="cleaned_video.mp4")
    parser.add_argument("-d", "--dir", help="path of the rejected frames dir", default="rejected_frames")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser

def load_frames_from_video(video_path, resize: Optional[Tuple]=None):
    """
    Reads all frames from the input video and returns them as a list.
    Parameter resize is useful to spare some ram but will impact the final result
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps

def write_video(frames, output_filename, fps):
    """
    Reconstructs a video from an ordered list of frames.
    """
    if not frames:
        print("No frames to write to video.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)
    video.release()
    print(f"Video saved as {output_filename}")


def write_rejected_frames(frames, directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{directory}/rejected_frame_{i}.jpg", frame)

def main():
    encoders = {
        'vit': VitEncoder,
        'orb': OrbEncoder
    }
    args = create_arg_parser().parse_args()
    if not args.filename:
        # I could also create my own derived exception but I don't think it's in scope for this test
        raise Exception("No filename provided")
    print(f"Loading video {args.filename}")
    frames, fps = load_frames_from_video(args.filename)
    if args.encoder not in ['vit', 'orb']:
        raise Exception(f"Unknown encoder {args.encoder}")
    print(f"Loading encoder {args.encoder}")
    encoder = encoders[args.encoder]()
    print("Extracting frames's features")
    features = np.array([encoder.encode(frame) for frame in frames])

    # Here I release the encoder so save ram
    del encoder


    print("Now filtering outlier frames")
    # First I filter the outlier frames
    frame_filter = FrameFilter(clustering_eps=0.5, filtering_threshold=2.5)
    filtered_frames_idx = frame_filter.filter_frames(features)

    # Now I order them
    filtered_frames_features = [feature for i, feature in enumerate(features) if i in filtered_frames_idx]
    filtered_frames = [frame for i, frame in enumerate(frames) if i in filtered_frames_idx]

    print("Now ordering remaining frames")
    frame_orderer = FrameOrderer()


    ordered_frames_idx = frame_orderer.order_frames(filtered_frames_features)

    # Here I save all
    print("All done ! Saving results")
    ordered_filtered_frames = [filtered_frames[i] for i in ordered_frames_idx]
    write_video(ordered_filtered_frames, args.output, fps)
    write_rejected_frames([frame for i, frame in enumerate(frames) if i not in filtered_frames_idx], args.dir)


if __name__ == '__main__':
    main()
