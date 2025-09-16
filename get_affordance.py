#!/usr/bin/env python3
"""
Combined script for loading images from VRS files and running hand-object detection.
Combines ProjectAria VRS data loading with Faster R-CNN hand-object detection.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from html import parser
import os
import sys
import csv
from matplotlib import pyplot as plt
import numpy as np
import argparse
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

# ProjectAria imports
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
from typing import Dict, List, Optional

from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from demo_video.predictor import VisualizationDemo

time_domain: TimeDomain = TimeDomain.DEVICE_TIME
time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST

NORMAL_VIS_LEN = 0.05  # meters

# Hand-object detection imports
import torchvision.transforms as transforms
import torchvision.datasets as dset

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

time_domain: TimeDomain = TimeDomain.DEVICE_TIME
time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST

# Helper functions for reprojection and plotting
def get_point_reprojection(
    T_world_device_proj, camera_calibrations, point_position_world: np.array, key: str = None
) -> Optional[np.array]:
    # online_camera_calibration = mps_data_provider.get_online_calibration(sample_timestamp_ns).get_camera_calib(f'camera-{key}')
    # online_camera_pose = online_camera_calibration.get_transform_device_camera()
    rgb_camera_calibration = camera_calibrations[key]
    T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
    T_world_camera = T_world_device_proj @ T_device_rgb_camera
    point_position_camera = T_world_camera.inverse() @ point_position_world
    point_position_pixel = camera_calibrations[key].project(point_position_camera)
    return point_position_pixel


def get_landmark_pixels(hand_tracking_result, key, T_world_device, T_world_device_proj, camera_calibrations) -> np.array:
    left_wrist = None
    left_palm = None
    left_landmarks = None
    right_wrist = None
    right_palm = None
    right_landmarks = None
    left_wrist_normal_tip = None
    left_palm_normal_tip = None
    right_wrist_normal_tip = None
    right_palm_normal_tip = None
    if hand_tracking_result.left_hand:
        left_landmarks = [
            get_point_reprojection(T_world_device_proj, camera_calibrations, T_world_device @ landmark, key)
            for landmark in hand_tracking_result.left_hand.landmark_positions_device
        ]
        left_landmarks = left_landmarks[:5]
        left_wrist = get_point_reprojection(
            T_world_device_proj, camera_calibrations, hand_tracking_result.left_hand.landmark_positions_device[
                int(mps.hand_tracking.HandLandmark.WRIST)
            ],
            key,
        )
        left_palm = get_point_reprojection(
            T_world_device_proj, camera_calibrations, hand_tracking_result.left_hand.landmark_positions_device[
                int(mps.hand_tracking.HandLandmark.PALM_CENTER)
            ],
            key,
        )
        if hand_tracking_result.left_hand.wrist_and_palm_normal_device is not None:
            left_wrist_normal_tip = get_point_reprojection(
                T_world_device_proj, camera_calibrations, hand_tracking_result.left_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.WRIST)
                ]
                + hand_tracking_result.left_hand.wrist_and_palm_normal_device.wrist_normal_device
                * NORMAL_VIS_LEN,
                key,
            )
            left_palm_normal_tip = get_point_reprojection(
                T_world_device_proj, camera_calibrations, hand_tracking_result.left_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                ]
                + hand_tracking_result.left_hand.wrist_and_palm_normal_device.palm_normal_device
                * NORMAL_VIS_LEN,
                key,
            )
    if hand_tracking_result.right_hand:
        right_landmarks = [
            get_point_reprojection(T_world_device_proj, camera_calibrations, T_world_device @ landmark, key)
            for landmark in hand_tracking_result.right_hand.landmark_positions_device
        ]
        right_landmarks = right_landmarks[:5]
        right_wrist = get_point_reprojection(
            T_world_device_proj, camera_calibrations, hand_tracking_result.right_hand.landmark_positions_device[
                int(mps.hand_tracking.HandLandmark.WRIST)
            ],
            key,
        )
        right_palm = get_point_reprojection(
            T_world_device_proj, camera_calibrations, hand_tracking_result.right_hand.landmark_positions_device[
                int(mps.hand_tracking.HandLandmark.PALM_CENTER)
            ],
            key,
        )
        if hand_tracking_result.right_hand.wrist_and_palm_normal_device is not None:
            right_wrist_normal_tip = get_point_reprojection(
                T_world_device_proj, camera_calibrations, hand_tracking_result.right_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.WRIST)
                ]
                + hand_tracking_result.right_hand.wrist_and_palm_normal_device.wrist_normal_device
                * NORMAL_VIS_LEN,
                key,
            )
            right_palm_normal_tip = get_point_reprojection(
                T_world_device_proj, camera_calibrations, hand_tracking_result.right_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                ]
                + hand_tracking_result.right_hand.wrist_and_palm_normal_device.palm_normal_device
                * NORMAL_VIS_LEN,
                key,
            )
    
    return (
        left_wrist,
        left_palm,
        right_wrist,
        right_palm,
        left_wrist_normal_tip,
        left_palm_normal_tip,
        right_wrist_normal_tip,
        right_palm_normal_tip,
        left_landmarks,
        right_landmarks
    )


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run hand-object detection on VRS file')
    
    # VRS file arguments
    parser.add_argument('--start_frame', dest='start_frame',
                        help='Starting frame index',
                        default=0, type=int)
    parser.add_argument('--end_frame', dest='end_frame',
                        help='Ending frame index (-1 for all frames)',
                        default=-1, type=int)
    parser.add_argument('--school', dest='school',
                        help='School name (e.g., "eth", "stanford")',
                        default='eth', type=str)
    parser.add_argument('--scene_idx', dest='scene_idx',
                        help='scene index',
                        default=-1, type=int)
    parser.add_argument('--recording_idx', dest='recording_idx',
                        help='recording index',
                        default=-1, type=int)
    parser.add_argument('--stream_type', dest='stream_type',
                        help='Stream type to use (rgb, slam-left, slam-right)',
                        default='rgb', type=str)
    
    args = parser.parse_args()
    return args


def setup_vrs_provider(vrs_file, stream_type='rgb'):
    """Setup VRS data provider and get stream information"""
    # Create data provider
    provider = data_provider.create_vrs_data_provider(vrs_file)
    
    # Define stream IDs based on stream type
    stream_id_map = {
        "rgb": StreamId("214-1"),
        "slam-left": StreamId("1201-1"),
        "slam-right": StreamId("1201-2"),
    }
    
    if stream_type not in stream_id_map:
        raise ValueError(f"Unknown stream type: {stream_type}. Available: {list(stream_id_map.keys())}")
    
    stream_id = stream_id_map[stream_type]
    time_domain = TimeDomain.DEVICE_TIME
    time_query_closest = TimeQueryOptions.CLOSEST
    
    # Get timestamps for the stream
    timestamps_ns = provider.get_timestamps_ns(stream_id, time_domain)
    
    print(f"Loaded VRS file: {vrs_file}")
    print(f"Stream type: {stream_type}")
    print(f"Total frames: {len(timestamps_ns)}")
    
    return provider, stream_id, timestamps_ns, time_domain, time_query_closest


def process_vrs_frame(provider, stream_id, timestamp_ns, time_domain, time_query_closest):
    """Load and process a single frame from VRS"""
    # Get image data
    image_data = provider.get_image_data_by_time_ns(
        stream_id, timestamp_ns, time_domain, time_query_closest
    )[0]
    
    # Convert to numpy array
    image_array = image_data.to_numpy_array()
    image_array = np.rot90(image_array, -1)
    
    # Convert RGB to BGR for OpenCV compatibility
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_array


def find_closest_future_contact_frame(frame_idx, contact_frames):
    """
    Find the closest future contact frame for a given frame index.
    
    Args:
        frame_idx: The current frame index
        contact_frames: List of frame indices where contact was detected
        
    Returns:
        The closest future contact frame index, or None if no future contact frame exists
    """
    if not contact_frames:
        return None
    
    # Sort contact frames to ensure they're in order
    sorted_contact_frames = sorted(contact_frames)
    
    # Find the first contact frame that comes after the current frame
    for contact_frame in sorted_contact_frames:
        if contact_frame > frame_idx:
            return contact_frame
    
    return None


def save_results_to_csv(mps_data_provider, mps_trajectory, camera_calibrations, detection_results, contact_frames, timestamps_ns, output_path):
    """
    Save detection results to CSV file.
    
    Args:
        detection_results: List of detection results for each frame
        intervals: List of consecutive contact intervals
        timestamps_ns: List of timestamps in nanoseconds
        output_path: Path to save the CSV file
    """
    
    # Prepare CSV data
    csv_data = []
    
    for result in detection_results:
        frame_idx = result['frame_idx']
        timestamp_ns = result['timestamp_ns']
        rgb_image = result['rgb']
        
        # Convert timestamp from nanoseconds to microseconds
        print("timestamp_ns", timestamp_ns)
        timestamp_us = timestamp_ns // 1000

        if frame_idx in contact_frames:
            is_contact = True
            hand_timestamps_ns = timestamps_ns[frame_idx]
            hand_tracking_result = mps_data_provider.get_hand_tracking_result(
                hand_timestamps_ns, time_query_closest
            )
        else:
            is_contact = False
            closest_future_frame = find_closest_future_contact_frame(frame_idx, contact_frames)
            if closest_future_frame is None:
                continue
            hand_timestamps_ns = timestamps_ns[closest_future_frame]
            hand_tracking_result = mps_data_provider.get_hand_tracking_result(
                hand_timestamps_ns, time_query_closest
            )

        pose_info = get_nearest_pose(mps_trajectory, hand_timestamps_ns)
        if pose_info is None:
            print(f"No pose found for timestamp {hand_timestamps_ns} in trajectory.")
            continue
        else:
            T_world_device = pose_info.transform_world_device

        pose_info = get_nearest_pose(mps_trajectory, timestamp_ns)
        if pose_info is None:
            print(f"No pose found for timestamp {timestamp_ns} in trajectory.")
            continue
        else:
            T_world_device_proj = pose_info.transform_world_device

        # (
        #     left_wrist,
        #     left_palm,
        #     right_wrist,
        #     right_palm,
        #     left_wrist_normal,
        #     left_palm_normal,
        #     right_wrist_normal,
        #     right_palm_normal,
        #     left_landmarks,
        #     right_landmarks,
        # ) = get_landmark_pixels(hand_tracking_result, "rgb", T_world_device, T_world_device_proj, camera_calibrations)

        if hand_tracking_result.left_hand:
            world_left_landmarks = [T_world_device @ landmark for landmark in hand_tracking_result.left_hand.landmark_positions_device]
            world_left_landmarks = world_left_landmarks[:5]
            left_contact_points_str = str(world_left_landmarks) if world_left_landmarks else "[]"
        if hand_tracking_result.right_hand:
            world_right_landmarks = [T_world_device @ landmark for landmark in hand_tracking_result.right_hand.landmark_positions_device]
            world_right_landmarks = world_right_landmarks[:5]
            right_contact_points_str = str(world_right_landmarks) if world_right_landmarks else "[]"

        # plt.figure()
        # plt.grid(False)
        # plt.axis("off")
        # plt.imshow(rgb_image)

        # plot_landmarks_and_connections(
        #     plt,
        #     left_landmarks,
        #     right_landmarks,
        #     mps.hand_tracking.kHandJointConnections,
        #     rgb_image.shape[0]
        # )

        # plt.savefig(f"/fsx-siro/liuzeyi/data/mps_object_in_container_eth_scene_{args.scene_idx}_recording_{args.recording_idx}_vrs/affordance/{frame_idx}.png", bbox_inches="tight")

        # print("left_landmarks", left_landmarks)
        # print("right_landmarks", right_landmarks)

        csv_data.append([timestamp_us, is_contact, left_contact_points_str, right_contact_points_str])

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['tracking_timestamp_us', 'contact', 'left_contact_points', 'right_contact_points'])
        # Write data
        writer.writerows(csv_data)
    
    print(f"Results saved to CSV: {output_path}")
    print(f"Total rows: {len(csv_data)}")
    print(f"Contact frames: {sum(1 for row in csv_data if row[1])}")

def load_detection_model(args):
    """Load the HOISTFormer detection model"""
    weights_path = './pretrained_models/trained_model.pth'
    cfg_file = './configs/hoist/hoistformer.yaml'
    
    demo_inference = VisualizationDemo(cfg_file, weights_path)
    return demo_inference

def run_detection_on_frame(demo_inference, frame):
    predictions, vis_output, instances = demo_inference.run_on_video(frame)
    return predictions, vis_output, instances


def main():
    args = parse_args()

    sample_path = "/fsx-siro/liuzeyi/data"
    mps_sample_path = f"/fsx-siro/liuzeyi/data/put_cup_on_saucer/mps_put_cup_on_saucer_{args.school}_scene_{args.scene_idx}_recording_{args.recording_idx}_vrs"

    # Load the VRS file
    vrs_file = os.path.join(sample_path, f"put_cup_on_saucer/put_cup_on_saucer_{args.school}_scene_{args.scene_idx}_recording_{args.recording_idx}.vrs")

    # Trajectory, global points, and online calibration
    closed_loop_trajectory = os.path.join(
        mps_sample_path, "slam", "closed_loop_trajectory.csv"
    )
    global_points = os.path.join(mps_sample_path, "slam", "semidense_points.csv.gz")

    # Hand tracking
    hand_tracking_results_path = os.path.join(
        mps_sample_path, "hand_tracking", "hand_tracking_results.csv"
    )
    # Create data provider and get T_device_rgb
    provider = data_provider.create_vrs_data_provider(vrs_file)
    device_calibration = provider.get_device_calibration()

    mps_data_provider = mps.MpsDataProvider(mps.MpsDataPathsProvider(mps_sample_path).get_data_paths())
    # Since we want to display the position of the RGB camera, we are querying its relative location
    # from the device and will apply it to the device trajectory.
    T_device_RGB = provider.get_device_calibration().get_transform_device_sensor(
        "camera-rgb"
    )
    stream_ids: Dict[str, StreamId] = {
        "rgb": StreamId("214-1"),
        "slam-left": StreamId("1201-1"),
        "slam-right": StreamId("1201-2"),
    }
    stream_labels: Dict[str, str] = {
        key: provider.get_label_from_stream_id(stream_id)
        for key, stream_id in stream_ids.items()
    }
    camera_calibrations = {
        key: device_calibration.get_camera_calib(stream_label)
        for key, stream_label in stream_labels.items()
    }

    rgb_camera_calibration = camera_calibrations['rgb']
    T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()

    ## Load trajectory and global points
    mps_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory)
    points = mps.read_global_point_cloud(global_points)

    hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
        hand_tracking_results_path
    )
    
    # Setup VRS provider
    provider, stream_id, timestamps_ns, time_domain, time_query_closest = setup_vrs_provider(
        vrs_file, args.stream_type
    )
    
    # Load detection model
    demo_inference = load_detection_model(args)
    
    # Determine frame range
    total_frames = 2000 # len(timestamps_ns)
    print("Total frames:", total_frames)
    
    # Storage for results
    contact_frames = []
    detection_results = []

    csv_output_path = f"/fsx-siro/liuzeyi/data/put_cup_on_saucer/mps_put_cup_on_saucer_{args.school}_scene_{args.scene_idx}_recording_{args.recording_idx}_vrs/contact_points.csv"

    images = []
    output_path = './output_results/'

    # if not os.path.exists(csv_output_path):
    if True:
        for frame_idx in range(total_frames):

            timestamp_ns = timestamps_ns[frame_idx]

            # Load frame from VRS
            im = process_vrs_frame(provider, stream_id, timestamp_ns, time_domain, time_query_closest)
            images.append(im)
            
            if len(images) == 10:
                # Run detection
                predictions, vis_output, instances = run_detection_on_frame(
                    demo_inference, np.array(images),
                )

                for idx, instance in enumerate(instances):
                    if np.any(np.array(instance.pred_masks)):
                        contact_detected = True
                    else:
                        contact_detected = False

                    # Store results
                    if contact_detected:
                        contact_frames.append(frame_idx - 9 + idx)
                    
                    detection_results.append({
                        'frame_idx': frame_idx - 9 + idx,
                        "rgb": images[idx],
                        'timestamp_ns': timestamps_ns[frame_idx - 9 + idx],
                        'contact_detected': contact_detected,
                    })

                # visualization
                frm = vis_output[-1].get_image()
                save_path = os.path.join(output_path, f'output_frm_{frame_idx}.jpg')
                cv2.imwrite(save_path, frm)
            
                images = []
                
                sys.stdout.write(f'Processed: {frame_idx} / {total_frames}\n')
                sys.stdout.flush()

                frame_idx += 1
        
        # Print results
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {len(detection_results)}")
        print(f"Frames with hand-object contact: {len(contact_frames)}")
        print(f"Contact frames: {sorted(contact_frames)}")
        
        # Save results summary
        # results_file = os.path.join(args.save_dir, "detection_results.txt")
        # with open(results_file, 'w') as f:
        #     f.write(f"VRS file: {args.vrs_file}\n")
        #     f.write(f"Stream type: {args.stream_type}\n")
        #     f.write(f"Frame range: {start_frame} to {end_frame}\n")
        #     f.write(f"Total frames processed: {len(detection_results)}\n")
        #     f.write(f"Frames with contact: {len(contact_frames)}\n")
        #     f.write(f"Contact frames: {sorted(contact_frames)}\n")
        #     f.write("\nDetailed results:\n")
        #     for result in detection_results:
        #         f.write(f"Frame {result['frame_idx']}: "
        #                f"Objects={result['obj_dets'] is not None}, "
        #                f"Hands={result['hand_dets'] is not None}, "
        #                f"Contact={result['contact_detected']}, "
        #                f"Time={result['detect_time']:.3f}s\n")
        
        # print(f"Results saved to: {results_file}")
        save_results_to_csv(mps_data_provider, mps_trajectory, camera_calibrations, detection_results, contact_frames, timestamps_ns, csv_output_path)
    else:
        print(f"{csv_output_path} already exists.")


if __name__ == '__main__':
    main()
    