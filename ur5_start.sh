roslaunch ur_modern_driver ur5_ros_control.launch robot_ip:=10.0.0.2
rosservice call /controller_manager/switch_controller "start_controllers: ['joint_group_vel_controller']
stop_controllers: ['vel_based_pos_traj_controller']
strictness: 0
start_asap: true 
timeout: 0.0" 
