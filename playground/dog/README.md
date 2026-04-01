# Dog 四足机器狗 - 使用说明

## 1. 项目概述

本项目是基于 `Open_Duck_Playground` 框架，为你的四足机器狗搭建的仿真训练和推理环境。机器人从 `my_assets/dog_urdf/urdf/dog.urdf` 转换为 MuJoCo XML 格式，并通过 Brax PPO 算法进行强化学习训练。

### 1.1 机器人参数


| 参数         | 值                   |
| ------------ | -------------------- |
| 机器人类型   | 四足机器狗           |
| 总质量       | ~2.14 kg             |
| 躯干质量     | 1.132 kg             |
| 腿部数量     | 4 (FR/FL/RR/RL)      |
| 每条腿关节数 | 3 (hip/thigh/calf)   |
| 总驱动关节数 | 12                   |
| 足部类型     | 球形碰撞体 (r=0.02m) |
| 站立高度     | ~0.30 m              |

### 1.2 关节说明

```
trunk (躯干)
├── FR_hip_joint  (前右髋关节, X轴旋转, ±0.80 rad)
│   └── FR_thigh_joint (前右大腿关节, Y轴旋转, -1.05~4.19 rad)
│       └── FR_calf_joint (前右小腿关节, Y轴旋转, -2.70~-0.92 rad)
│           └── FR_foot (前右足, 固连)
├── FL_hip_joint  (前左髋关节)
│   └── FL_thigh_joint → FL_calf_joint → FL_foot
├── RR_hip_joint  (后右髋关节)
│   └── RR_thigh_joint → RR_calf_joint → RR_foot
└── RL_hip_joint  (后左髋关节)
    └── RL_thigh_joint → RL_calf_joint → RL_foot
```

## 2. 目录结构

```
playground/dog/
├── __init__.py              # Python 包标记
├── constants.py             # 机器人常量：XML路径、关节名、传感器名等
├── base.py                  # DogEnv 基类：加载模型、关节索引映射、传感器读取
├── joystick.py              # Joystick 行走任务环境（速度追踪）
├── standing.py              # Standing 站立平衡任务环境
├── runner.py                # 训练入口：DogRunner
├── mujoco_infer_base.py     # MuJoCo CPU 推理基类（NumPy）
├── mujoco_infer.py          # ONNX 策略推理 + MuJoCo Viewer 可视化
├── README.md                # 本使用说明
└── xmls/
    ├── dog.xml              # 主机器人 MuJoCo 模型
    ├── joints_properties.xml # 关节电机属性默认值
    ├── sensors.xml          # IMU + 足部传感器定义
    ├── scene_flat_terrain.xml # 平坦地形场景
    └── assets/              # 17 个 STL 网格文件
        ├── trunk.STL
        ├── FR_hip.STL / FL_hip.STL / RR_hip.STL / RL_hip.STL
        ├── FR_thigh.STL / FL_thigh.STL / RR_thigh.STL / RL_thigh.STL
        ├── FR_calf.STL / FL_calf.STL / RR_calf.STL / RL_calf.STL
        └── FR_foot.STL / FL_foot.STL / RR_foot.STL / RL_foot.STL
```

## 3. 训练

### 3.1 Joystick 行走训练（推荐先训练这个）

```bash
python -m playground.dog.runner \
    --env joystick \
    --task flat_terrain \
    --output_dir checkpoints/dog_joystick \
    --num_timesteps 1500000
```

#### 训练参数说明


| 参数                        | 默认值         | 说明                                              |
| --------------------------- | -------------- | ------------------------------------------------- |
| `--env`                     | `joystick`     | 环境类型：`joystick`（行走）或 `standing`（站立） |
| `--task`                    | `flat_terrain` | 地形类型                                          |
| `--output_dir`              | `checkpoints`  | checkpoint 和 ONNX 模型的保存目录                 |
| `--num_timesteps`           | `150000000`    | 总训练步数（1.5亿步约需数小时，视 GPU 而定）      |
| `--restore_checkpoint_path` | `None`         | 从已有 checkpoint 恢复训练                        |

#### 行走任务奖励设计


| 奖励/惩罚          | 权重   | 说明                             |
| ------------------ | ------ | -------------------------------- |
| `tracking_lin_vel` | +2.5   | 线速度追踪奖励                   |
| `tracking_ang_vel` | +6.0   | 角速度追踪奖励                   |
| `alive`            | +20.0  | 存活奖励                         |
| `feet_air_time`    | +1.0   | 脚部抬起时间奖励（鼓励交替迈步） |
| `orientation`      | -0.5   | 姿态倾斜惩罚                     |
| `torques`          | -0.001 | 力矩惩罚（节省能量）             |
| `action_rate`      | -0.5   | 动作变化率惩罚（平滑运动）       |
| `stand_still`      | -0.2   | 静止时偏离默认姿态的惩罚         |

#### 观测空间（85维）

```
gyro(3) + accelerometer(3) + command(3) +
joint_angles(12) + joint_vel(12) +
last_action(12) + last_last_action(12) + last_last_last_action(12) +
motor_targets(12) + foot_contacts(4)
= 85 维
```

#### 命令空间（3维）

```
[lin_vel_x, lin_vel_y, ang_vel_yaw]
  范围: [-0.3, 0.3], [-0.2, 0.2], [-1.0, 1.0]
```

### 3.2 Standing 站立训练

```bash
python -m playground.dog.runner \
    --env standing \
    --task flat_terrain \
    --output_dir checkpoints/dog_standing \
    --num_timesteps 1500000
```

### 3.3 从 Checkpoint 恢复训练

```bash
python -m playground.dog.runner \
    --env joystick \
    --task flat_terrain \
    --output_dir checkpoints/dog_joystick \
    --restore_checkpoint_path checkpoints/dog_joystick/2026_04_01_120000_50000000
```

### 3.4 监控训练（TensorBoard）

```bash
tensorboard --logdir checkpoints/dog_joystick
```

在浏览器中打开 `http://localhost:6006` 可以查看：

- `eval/episode_reward`: 评估回合总奖励（核心指标）
- `reward/tracking_lin_vel`: 线速度追踪效果
- `cost/torques`: 力矩消耗
- 其他各项奖励/惩罚分量

## 4. 推理与可视化

训练完成后会自动导出 ONNX 模型。使用以下命令在 MuJoCo Viewer 中可视化推理效果：

```bash
python -m playground.dog.mujoco_infer \
    -o checkpoints/dog_joystick/2026_04_01_194817_50462720.onnx \
    --model_path playground/dog/xmls/scene_flat_terrain.xml
```

### 4.1 键盘控制


| 按键        | 功能       |
| ----------- | ---------- |
| ↑ (上箭头) | 向前移动   |
| ↓ (下箭头) | 向后移动   |
| ← (左箭头) | 向左平移   |
| → (右箭头) | 向右平移   |
| Q           | 逆时针旋转 |
| E           | 顺时针旋转 |

松开按键后速度命令归零，机器狗停止移动。

## 5. 核心文件说明

### 5.1 `constants.py`

定义所有与机器狗相关的常量：

- XML 文件路径
- 关节名列表（`JOINTS_ORDER`）
- 足部 site/geom 名
- 传感器名

如需修改关节顺序或增加地形，在此文件中修改。

### 5.2 `base.py`

`DogEnv` 基类，封装了：

- MuJoCo 模型加载和 MJX 加速
- 浮动基座 / 驱动关节 / backlash 关节的索引管理
- `get_gyro()`, `get_accelerometer()`, `get_feet_pos()` 等传感器读取方法
- `get_actuator_joints_qpos()`, `set_actuator_joints_qpos()` 等关节状态操作

### 5.3 `joystick.py`

行走任务环境（继承 `DogEnv`）：

- `reset()`: 初始化姿态 + 随机扰动
- `step()`: 接收动作、施加推力扰动、计算奖励
- `sample_command()`: 随机采样速度命令
- 支持动作延迟、IMU延迟、推力扰动等 sim2real 技巧

### 5.4 `standing.py`

站立平衡任务环境：

- 命令恒为零（不需要运动）
- 奖励重点在姿态保持和抗扰动

### 5.5 `runner.py`

训练入口，连接环境和 PPO 训练循环。

### 5.6 `mujoco_infer.py`

ONNX 模型推理可视化工具，支持键盘控制。

## 6. 自定义与调参指南

### 6.1 调整奖励权重

编辑 `joystick.py` 中的 `default_config()` 函数：

```python
reward_config=config_dict.create(
    scales=config_dict.create(
        tracking_lin_vel=2.5,   # 增大 → 更紧密追踪速度命令
        tracking_ang_vel=6.0,   # 增大 → 更紧密追踪转向命令
        orientation=-0.5,       # 增大绝对值 → 更严格保持水平
        torques=-1.0e-3,        # 增大绝对值 → 更节能但可能动力不足
        action_rate=-0.5,       # 增大绝对值 → 动作更平滑但响应变慢
        stand_still=-0.2,       # 增大绝对值 → 静止时更严格回到默认姿态
        alive=20.0,             # 存活奖励，保持较高值
        feet_air_time=1.0,      # 增大 → 更鼓励抬腿迈步
    ),
),
```

### 6.2 调整速度范围

编辑 `joystick.py` 中的 `default_config()`：

```python
lin_vel_x=[-0.3, 0.3],     # 前后速度范围 (m/s)
lin_vel_y=[-0.2, 0.2],     # 左右速度范围 (m/s)
ang_vel_yaw=[-1.0, 1.0],   # 旋转速度范围 (rad/s)
```

### 6.3 调整噪声

降低噪声可加速训练，但可能降低 sim2real 效果：

```python
noise_config=config_dict.create(
    level=1.0,              # 0.0 = 无噪声，1.0 = 全噪声
    action_min_delay=0,     # 最小动作延迟（控制步数）
    action_max_delay=3,     # 最大动作延迟
),
```

### 6.4 调整电机参数

编辑 `xmls/dog.xml` 中的 `dog_motor` default class：

```xml
<default class="dog_motor">
    <joint damping="0.60" frictionloss="0.052" armature="0.028"/>
    <position kp="17.8" kv="0.0" forcerange="-3.35 3.35"/>
</default>
```

- `kp`: 位置控制增益（增大 → 关节更刚性）
- `forcerange`: 最大力矩限制（根据你的电机规格调整）
- `damping`: 关节阻尼
- `frictionloss`: 静摩擦力矩

### 6.5 添加粗糙地形

1. 在 `xmls/` 下创建 `scene_rough_terrain.xml`（参考 open_duck_mini_v2 的同名文件）
2. 在 `constants.py` 中添加路径和映射：
   ```python
   ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_rough_terrain.xml"
   ```
3. 更新 `task_to_xml()` 函数

## 7. Sim2Real 迁移注意事项

1. **电机参数校准**: 确保 `dog.xml` 中的 `kp`, `damping`, `forcerange` 与真实电机匹配
2. **延迟模拟**: 训练时默认启用了 0-3 步的动作延迟和 IMU 延迟
3. **域随机化**: 训练自动包含地面摩擦、质量、关节参数的随机化
4. **推力扰动**: 训练时会随机施加外力推动，提高鲁棒性
5. **动作平滑**: 电机速度限制（5.24 rad/s）在训练和推理中都启用

## 8. 常见问题

### Q: 训练一直摔倒？

- 检查 `alive` 奖励是否足够高
- 降低 `action_scale`（默认 0.25）
- 确认 `scene_flat_terrain.xml` 中初始高度足够
- 降低 `push_config.magnitude_range`

### Q: 训练收敛但步态不自然？

- 增大 `action_rate` 惩罚的绝对值
- 增大 `feet_air_time` 奖励权重
- 增大 `torques` 惩罚

### Q: 训练太慢？

- 减小 `num_timesteps` 先快速验证
- 确认使用了 GPU（Brax/JAX 自动检测）
- 降低 `noise_config.level` 加速前期收敛

### Q: ONNX 导出失败？

- 确认安装了 `tensorflow`, `tf2onnx`, `onnxruntime`
- 检查 `playground/common/export_onnx.py` 的网络层配置
