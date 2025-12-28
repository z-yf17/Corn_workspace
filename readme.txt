###### ZMQ 发布相机 rgbd 信息流 ######
cd /home/galbot/ros_noetic_docker/Grounded-SAM-2
python host_video_publisher_rgbd.py


###### 打开 Grounded SAM V2 并发布分割结果的信息流 ######
进入docker Corn_docker
docker start Corn_docker
conda activate sam
cd /mnt/data/Grounded-SAM-2
python docker_video_sub_pub_tracking_rgb.py


###### 打开订阅并可视化分割结果 ######
cd /home/galbot/ros_noetic_docker/Grounded-SAM-2
python host_video_subscriber_rgbd_filter.py



###### Foundation Pose ######
cd /home/galbot/ros_noetic_docker/FoundationPose
bash docker/run_container.sh


###### Corn ######
PYMESHLAB_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYMESHLAB_LIB="$PYMESHLAB_SITE/pymeshlab/lib"
export LD_LIBRARY_PATH="$PYMESHLAB_LIB:$CONDA_PREFIX/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
python -m pip install -U "hydra-core==1.3.2" "omegaconf==2.3.0"


###### FoundationPose ######
cd /home/galbot/ros_noetic_docker/FoundationPose
python realtime.py


###### calibration ######






(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ nmcli connection show
NAME             UUID                                  TYPE      DEVICE          
netplan-enp4s0   ce7986cc-a2b9-4edb-a8d5-df0f4abd710f  ethernet  enp4s0          
br-d8578df619d2  3aece5f9-6ba4-4270-b03b-d3183b49780a  bridge    br-d8578df619d2 
docker0          996c122f-a06b-4775-9749-575bac23fec5  bridge    docker0         
br-237b82586074  48ef30f0-c360-4645-a254-9d2914c47304  bridge    br-237b82586074 
br-d98178108b41  1813badb-9b8f-4ae5-b858-7056c52e7b56  bridge    br-d98178108b41 
GALBOT-10F       cd7a5b67-50d9-441b-bc17-c0902c3afd87  wifi      --              
Jixiebi          12b9b8ae-d26a-4cb4-8043-a38f5d44ba96  ethernet  --              
Profile 1        8197903c-9bfd-4e7f-b904-6fa1dbbc817c  ethernet  --              
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ ping 192.168.1.10

PING 192.168.1.10 (192.168.1.10) 56(84) bytes of data.
^C
--- 192.168.1.10 ping statistics ---
3 packets transmitted, 0 received, 100% packet loss, time 2036ms

(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ sudo cat /etc/NetworkManager/system-connections/netplan-enp4s0.nmconnection
[connection]
id=netplan-enp4s0
uuid=ce7986cc-a2b9-4edb-a8d5-df0f4abd710f
type=ethernet
interface-name=enp4s0
permissions=

[ethernet]
mac-address-blacklist=

[802-1x]
eap=peap;
identity=18382277018
password=ZWEFbmGVMC
phase2-auth=mschapv2

[ipv4]
dns-search=
method=auto

[ipv6]
addr-gen-mode=stable-privacy
dns-search=
method=auto

[proxy]
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ 
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ nmcli connection show netplan-enp4s0 | grep ipv4.never-default 
ipv4.never-default:                     no
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ 
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ 
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ 
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ sudo nmcli connection modify netplan-enp4s0 +ipv4.addresses 192.168.0.100/24
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ 
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ ping 192.168.1.10

PING 192.168.1.10 (192.168.1.10) 56(84) bytes of data.
^C
--- 192.168.1.10 ping statistics ---
3 packets transmitted, 0 received, 100% packet loss, time 2033ms

(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ nmcli connection show netplan-enp4s0 | grep ipv4.addresses 
ipv4.addresses:                         192.168.0.100/24
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ sudo cat /etc/NetworkManager/system-connections/netplan-enp4s0.nmconnection
[connection]
id=netplan-enp4s0
uuid=ce7986cc-a2b9-4edb-a8d5-df0f4abd710f
type=ethernet
interface-name=enp4s0
permissions=
timestamp=1766140008

[ethernet]
mac-address-blacklist=

[802-1x]
eap=peap;
identity=18382277018
password=ZWEFbmGVMC
phase2-auth=mschapv2

[ipv4]
address1=192.168.0.100/24
dns-search=
method=auto

[ipv6]
addr-gen-mode=stable-privacy
dns-search=
method=auto

[proxy]
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ sudo nmcli connection down netplan-enp4s0\nsudo nmcli connection up netplan-enp4s0
Error: 'netplan-enp4s0nsudo' is not an active connection.
Error: 'nmcli' is not an active connection.
Error: 'connection' is not an active connection.
Error: 'up' is not an active connection.
Connection 'netplan-enp4s0' successfully deactivated (D-Bus active path: /org/freedesktop/NetworkManager/ActiveConnection/12)
Error: not all active connections found.
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ 
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ 
(base) ➜  /home/galbot/ros_noetic_docker git:(main) ✗ ping 192.168.1.10                                         


