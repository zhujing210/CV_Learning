**千万不要直接百度ros安装方法，那很大概率会卡在ros update这一步！！！！！！！！！！坑了我两天！！**



**正确办法：**

TX2上的ROS自动安装git上有现成的，记录如下：
首先 下载git上的自动安装脚本

git clone https://github.com/jetsonhacks/installROSTX2.git

其次 安装全量ros版本

> ./installROS.sh -p ros-kinetic-desktop-full

后面如果不给-p ros-kinetic-desktop-full，系统会默认安装ros-kinetic-ros-base

安装一些依赖:`sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential`

安装过程需要保证网络，过程可能会有些久，并且需要经常输入sudo密码。

然后`source ~/.bashrc`