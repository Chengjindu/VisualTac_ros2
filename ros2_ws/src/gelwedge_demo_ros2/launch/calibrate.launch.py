from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

def generate_launch_description():
    # Arguments to choose camera stream type and params
    stream_type = LaunchConfiguration('stream_type')
    mjpg_url    = LaunchConfiguration('mjpg_url')
    gst_port    = LaunchConfiguration('gst_port')

    declare_stream_type = DeclareLaunchArgument(
        'stream_type',
        default_value='mjpg',
        description='Camera stream type: "mjpg" or "gst"'
    )

    declare_mjpg_url = DeclareLaunchArgument(
        'mjpg_url',
        default_value='',
        description='MJPG stream URL, e.g., http://<pi-ip>:8080/?action=stream'
    )

    declare_gst_port = DeclareLaunchArgument(
        'gst_port',
        default_value='5000',
        description='GStreamer UDP port if using gst/gs source'
    )

    # We launch the calibration utility (non-ROS Python) via ExecuteProcess.
    # It expects:
    #   python -m gelwedge_demo_ros2.transformation_matrix_calculation --source mjpg --url <...>
    # or:
    #   python -m gelwedge_demo_ros2.transformation_matrix_calculation --source gs --port <...>
    run_mjpg = ExecuteProcess(
        cmd=[
            'python3', '-m', 'gelwedge_demo_ros2.transformation_matrix_calculation',
            '--source', 'mjpg',
            '--url', mjpg_url
        ],
        output='screen',
        shell=False,
        condition=IfCondition(
            PythonExpression(['"', stream_type, '" == "mjpg"'])
        )
    )

    run_gst = ExecuteProcess(
        cmd=[
            'python3', '-m', 'gelwedge_demo_ros2.transformation_matrix_calculation',
            '--source', 'gs',
            '--port', gst_port
        ],
        output='screen',
        shell=False,
        condition=IfCondition(
            PythonExpression(['"', stream_type, '" != "mjpg"'])
        )
    )

    return LaunchDescription([
        declare_stream_type,
        declare_mjpg_url,
        declare_gst_port,
        run_mjpg,
        run_gst,
    ])
