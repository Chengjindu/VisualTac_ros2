#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Launch arguments ----
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='mjpg',  # 'gs' or 'mjpg'
        description='Select camera source: "gs" for GStreamer UDP or "mjpg" for MJPG-streamer',
    )

    mjpg_url_arg = DeclareLaunchArgument(
        'mjpg_url',
        default_value='http://172.17.167.138:8080//?action=stream',
        description='URL for MJPG-streamer source (used if camera_type=mjpg)',
    )

    gst_port_arg = DeclareLaunchArgument(
        'gst_port',
        default_value='5000',
        description='UDP port for GStreamer video stream (used if camera_type=gs)',
    )

    frame_width_arg = DeclareLaunchArgument(
        'frame_width',
        default_value='800',
        description='Frame width after warp',
    )

    frame_height_arg = DeclareLaunchArgument(
        'frame_height',
        default_value='600',
        description='Frame height after warp',
    )

    # ---- Node definition ----
    node = Node(
        package='gelwedge_demo_ros2',
        executable='gelwedge_demo_node',
        name='gelwedge_demo',
        output='screen',
        parameters=[{
            'camera_type': LaunchConfiguration('camera_type'),
            'mjpg_url': LaunchConfiguration('mjpg_url'),
            'gst_port': LaunchConfiguration('gst_port'),
            'frame_width': LaunchConfiguration('frame_width'),
            'frame_height': LaunchConfiguration('frame_height'),
        }]
    )

    return LaunchDescription([
        camera_type_arg,
        mjpg_url_arg,
        gst_port_arg,
        frame_width_arg,
        frame_height_arg,
        node
    ])
