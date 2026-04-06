# Open Duck Playground 项目说明

## 1. 项目简介

`open_duck_playground` 是一个面向 **Open Duck Mini V2** 机器人步态与控制策略开发的强化学习实验项目。它基于 **MuJoCo / MJX + JAX + Brax PPO** 的技术栈，提供了从环境定义、奖励函数设计、训练、模型导出到 MuJoCo 推理验证的一整套实验流程。

当前仓库的核心目标是：

- 为 Open Duck Mini V2 构建可训练的仿真环境；
- 训练站立、行走等控制策略；
- 支持加入 imitation reward（模仿奖励）以利用参考运动；
- 将训练得到的策略导出为 ONNX 模型，便于后续部署与推理；
- 在 MuJoCo 中进行交互式推理和控制验证。

从现有代码看，这个仓库已经不仅是一个“示例 playground”，而是一个围绕具体机器人平台组织起来的训练与验证工程。

---

## 2. 技术栈

项目主要依赖如下：

- **Python 3.11+**
- **JAX / jaxlib**：用于高性能数值计算与 MJX 仿真
- **MuJoCo / MJX**：物理仿真与环境执行
- **Brax PPO**：强化学习训练算法
- **TensorFlow + tf2onnx**：用于将策略导出为 ONNX
- **onnxruntime**：用于 ONNX 模型推理
- **matplotlib / mediapy / pygame**：用于可视化、交互与辅助工具
- **uv**：推荐的 Python 包与运行环境管理工具

项目在 [pyproject.toml](pyproject.toml) 中声明了这些依赖。

---

## 3. 项目定位与工作流

这个仓库围绕一个典型的机器人策略研发闭环展开：

1. **定义机器人环境**
   - 在机器人目录中提供 MuJoCo XML、常量配置、基础环境类与任务环境。

2. **设计观测、动作与奖励**
   - 在任务文件中定义状态观测、命令空间、奖励项、噪声模型与随机化逻辑。

3. **训练 PPO 策略**
   - 通过统一 runner 启动训练，记录 TensorBoard 指标，并定期保存 checkpoint。

4. **导出 ONNX 模型**
   - 训练过程中自动将策略网络导出为 ONNX，便于脱离训练框架进行推理。

5. **在 MuJoCo 中推理验证**
   - 用导出的 ONNX 模型驱动机器人，在仿真中通过键盘交互测试控制效果。

这使得仓库既适合做强化学习研究，也适合做具体机器人平台上的控制策略迭代。

---

## 4. 目录结构

仓库当前的核心结构如下：

```text
.
├── README.md
├── pyproject.toml
├── playground/
│   ├── common/
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── plot_saved_obs.py
│   │   ├── poly_reference_motion.py
│   │   ├── poly_reference_motion_numpy.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   ├── rewards_numpy.py
│   │   ├── runner.py
│   │   └── utils.py
│   └── open_duck_mini_v2/
│       ├── __init__.py
│       ├── base.py
│       ├── constants.py
│       ├── custom_rewards.py
│       ├── custom_rewards_numpy.py
│       ├── joystick.py
│       ├── mujoco_infer.py
│       ├── mujoco_infer_base.py
│       ├── ref_motion_viewer.py
│       ├── runner.py
│       ├── standing.py
│       ├── data/
│       │   └── polynomial_coefficients.pkl
│       └── xmls/
│           ├── assets/
│           └── *.xml
```

### 4.1 `playground/common/`

这一层是跨机器人或跨任务的公共能力：

- [playground/common/runner.py](playground/common/runner.py)
  - 统一训练入口基类；
  - 对接 Brax PPO；
  - 记录 TensorBoard；
  - 保存 checkpoint；
  - 自动导出 ONNX。

- [playground/common/export_onnx.py](playground/common/export_onnx.py)
  - 将训练好的策略参数转换成 TensorFlow/Keras 模型；
  - 再导出为 ONNX，用于后续部署与推理。

- [playground/common/onnx_infer.py](playground/common/onnx_infer.py)
  - 提供 ONNX 模型推理封装。

- [playground/common/rewards.py](playground/common/rewards.py)
  - 公共奖励函数实现，如速度跟踪、力矩代价、动作变化代价等。

- [playground/common/randomize.py](playground/common/randomize.py)
  - 域随机化逻辑，用于提升策略鲁棒性。

- [playground/common/poly_reference_motion.py](playground/common/poly_reference_motion.py)
  - 读取并生成多项式参考动作，用于 imitation reward。

- [playground/common/poly_reference_motion_numpy.py](playground/common/poly_reference_motion_numpy.py)
  - 上述逻辑的 NumPy 推理版本，便于推理侧使用。

### 4.2 `playground/open_duck_mini_v2/`

这是当前仓库最核心的机器人实现目录，对应 Open Duck Mini V2。

- [playground/open_duck_mini_v2/base.py](playground/open_duck_mini_v2/base.py)
  - Open Duck Mini V2 的基础环境类；
  - 负责加载 MuJoCo XML 与 assets；
  - 建立 actuator、joint、floating base、sensor 等索引；
  - 提供一组用于读取/设置机器人状态的基础方法。

- [playground/open_duck_mini_v2/constants.py](playground/open_duck_mini_v2/constants.py)
  - 存放机器人关键命名常量，例如 body、site、sensor、geom、joint 顺序等。

- [playground/open_duck_mini_v2/joystick.py](playground/open_duck_mini_v2/joystick.py)
  - 行走/速度跟踪任务；
  - 定义环境默认参数、奖励配置、命令采样、观测构造与 reset/step 行为；
  - 支持 imitation reward。

- [playground/open_duck_mini_v2/standing.py](playground/open_duck_mini_v2/standing.py)
  - 站立任务环境。

- [playground/open_duck_mini_v2/custom_rewards.py](playground/open_duck_mini_v2/custom_rewards.py)
  - Open Duck Mini V2 专用奖励项，例如 imitation 相关奖励。

- [playground/open_duck_mini_v2/runner.py](playground/open_duck_mini_v2/runner.py)
  - Open Duck Mini V2 的训练入口；
  - 当前支持 `joystick` 和 `standing` 两类环境。

- [playground/open_duck_mini_v2/mujoco_infer.py](playground/open_duck_mini_v2/mujoco_infer.py)
  - 使用导出的 ONNX 模型在 MuJoCo 中执行交互式推理；
  - 支持键盘输入速度命令与头部控制。

- [playground/open_duck_mini_v2/mujoco_infer_base.py](playground/open_duck_mini_v2/mujoco_infer_base.py)
  - MuJoCo 推理运行时所需的底层封装。

- [playground/open_duck_mini_v2/ref_motion_viewer.py](playground/open_duck_mini_v2/ref_motion_viewer.py)
  - 参考运动查看工具。

- `data/polynomial_coefficients.pkl`
  - 模仿奖励所需的参考运动多项式系数。

- `xmls/` 和 `xmls/assets/`
  - 机器人 MuJoCo 模型、场景配置与 CAD/网格资源。

---

## 5. 核心模块说明

### 5.1 训练系统

训练逻辑以 [playground/common/runner.py](playground/common/runner.py) 为公共基类，由各机器人 runner 派生实现。

以 [playground/open_duck_mini_v2/runner.py](playground/open_duck_mini_v2/runner.py) 为例：

- 根据 `--env` 选择任务环境；
- 创建训练环境与评估环境；
- 接入域随机化；
- 计算 observation/action 维度；
- 调用公共 `train()` 方法启动 Brax PPO 训练。

训练过程中，系统会：

- 输出评估 reward；
- 将指标写入 TensorBoard；
- 定期保存参数 checkpoint；
- 自动导出 ONNX 模型。

### 5.2 机器人环境基类

[playground/open_duck_mini_v2/base.py](playground/open_duck_mini_v2/base.py) 是机器人环境的底座，负责：

- 加载 XML 模型与资源；
- 创建 `MjModel` 和 `mjx` 模型；
- 识别 floating base、actuator joints、backlash joints；
- 提供 joint/qpos/qvel 的访问与更新函数；
- 封装传感器读取逻辑。

这意味着上层任务文件可以把重点放在：

- 任务目标；
- 奖励设计；
- 观测设计；
- reset 与 step 策略。

### 5.3 任务环境

当前至少包含两类任务：

- `joystick`：更偏向速度跟踪、行走控制；
- `standing`：更偏向静态站立稳定性。

其中 [playground/open_duck_mini_v2/joystick.py](playground/open_duck_mini_v2/joystick.py) 是最典型的任务文件，包含：

- 控制步长、仿真步长、episode 长度；
- 动作缩放与关节速度缩放；
- 噪声配置；
- 奖励权重；
- 外力 push 配置；
- 线速度/角速度/头部动作命令范围；
- imitation reward 开关。

这个文件基本体现了项目中“训练一个行走策略”所需的大部分关键设计。

### 5.4 imitation reward

项目支持通过参考运动来增强训练。现有 README 已说明：

- 可先在外部参考运动生成仓库中生成运动轨迹；
- 然后将 `polynomial_coefficients.pkl` 放到对应机器人目录下的 `data/` 中；
- 在任务文件中启用 `USE_IMITATION_REWARD=True`。

在当前实现中：

- JAX 训练侧使用 [playground/common/poly_reference_motion.py](playground/common/poly_reference_motion.py)；
- NumPy / 推理侧使用 [playground/common/poly_reference_motion_numpy.py](playground/common/poly_reference_motion_numpy.py)；
- 机器人特定 imitation 奖励位于 [playground/open_duck_mini_v2/custom_rewards.py](playground/open_duck_mini_v2/custom_rewards.py)。

### 5.5 ONNX 导出与推理

[playground/common/export_onnx.py](playground/common/export_onnx.py) 会在训练期间把 PPO 策略参数转换为一个 TensorFlow 模型，并进一步导出 ONNX。

然后可通过 [playground/open_duck_mini_v2/mujoco_infer.py](playground/open_duck_mini_v2/mujoco_infer.py) 加载 ONNX 模型做 MuJoCo 推理测试。

该推理脚本的主要用途是：

- 验证训练策略是否可稳定执行；
- 交互式测试速度指令；
- 在站立/行走模式下检查策略行为；
- 保存推理时的观测供后续分析。

---

## 6. 如何安装

项目 README 推荐使用 `uv`。

### 6.1 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 6.2 安装项目依赖

如果本地已安装 `uv`，通常可以在仓库根目录执行：

```bash
uv sync
```

如果你更习惯直接基于 `pyproject.toml` 安装，也可以按自己的 Python 环境方式处理。

---

## 7. 如何训练

当前主要训练入口是 Open Duck Mini V2 的 runner：

```bash
uv run playground/open_duck_mini_v2/runner.py
```

常用参数包括：

- `--output_dir`：checkpoint 和日志输出目录
- `--num_timesteps`：训练步数
- `--env`：环境类型，当前代码中支持 `joystick` / `standing`
- `--task`：具体场景任务
- `--restore_checkpoint_path`：从已有 checkpoint 恢复训练

示例：

```bash
uv run playground/open_duck_mini_v2/runner.py --env joystick --task flat_terrain
```

README 中还给出过一个较长训练示例：

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

训练时建议同时打开 TensorBoard 观察指标。

---

## 8. 如何查看训练日志

```bash
uv run tensorboard --logdir=<yourlogdir>
```

日志由公共 runner 写入输出目录，可用于观察：

- episode reward
- eval reward
- 各奖励项/代价项变化
- 训练过程中的稳定性趋势

---

## 9. 如何进行 MuJoCo 推理

当前仓库已经提供面向 Open Duck Mini V2 的推理脚本：

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_onnx_model>
```

常见参数包括：

- `-o / --onnx_model_path`：必填，ONNX 模型路径
- `--reference_data`：参考运动数据路径
- `--model_path`：MuJoCo XML 场景路径
- `--standing`：切换为站立模式

该脚本会：

- 打开 MuJoCo 可视化窗口；
- 读取键盘输入生成命令；
- 构造观测；
- 调用 ONNX policy 推理动作；
- 将动作写回 MuJoCo 控制器。

---

## 10. 如何添加新机器人

README 中已经给出了扩展思路。若要在该框架下新增机器人，通常可按如下方式进行：

1. 在 `playground/` 下创建新的机器人目录；
2. 参考 `open_duck_mini_v2/` 的结构复制一份模板；
3. 修改 `base.py`
   - 适配机器人 XML、关节、执行器与状态读取逻辑；
4. 修改 `constants.py`
   - 定义关键 body、geom、site、sensor、joint 常量；
5. 放入机器人自己的 `xmls/` 与 `assets/`；
6. 编写任务文件，如 `joystick.py` / `standing.py`
   - 设计奖励函数、观测、命令空间与 reset 逻辑；
7. 编写 `runner.py`
   - 将任务注册到训练入口；
8. 如需 imitation reward，再准备对应参考运动数据。

从现有项目结构看，Open Duck Mini V2 已经是一个可复用模板。

---

## 11. 当前项目特点总结

这个仓库的几个特点比较明确：

- **面向真实机器人平台组织**：不是泛化 benchmark，而是围绕 Open Duck Mini V2 展开；
- **训练与部署链路完整**：从环境、训练到 ONNX 推理验证都在仓库中；
- **支持 imitation reward**：可结合外部参考运动提升策略质量；
- **强调 MuJoCo/MJX 与 JAX 生态**：训练计算与仿真链路现代化；
- **具备扩展为多机器人框架的潜力**：`common/` 与机器人目录分层已经形成。

---

## 12. 适合的使用场景

这个项目适合以下工作：

- Open Duck Mini V2 的步态训练与实验；
- 机器人站立/行走控制策略调参；
- 使用 imitation reward 的参考运动学习；
- 将强化学习策略导出为 ONNX 并做部署前验证；
- 作为新增机器人训练目录的模板工程。

---

## 13. 已知现状与阅读建议

从当前仓库内容看，有几点需要注意：

- README 仍较简略，部分描述还停留在早期阶段；
- 项目实际功能已比 README 展示的更多，例如：
  - `standing` 环境；
  - `mujoco_infer_base.py`；
  - `ref_motion_viewer.py`；
  - NumPy 版本 reward / reference motion；
- 一些默认参数和任务命名可能仍在快速迭代中，建议以源码实现为准。

如果你是第一次接触该项目，推荐按下面顺序阅读：

1. [README.md](README.md)
2. [pyproject.toml](pyproject.toml)
3. [playground/common/runner.py](playground/common/runner.py)
4. [playground/open_duck_mini_v2/runner.py](playground/open_duck_mini_v2/runner.py)
5. [playground/open_duck_mini_v2/base.py](playground/open_duck_mini_v2/base.py)
6. [playground/open_duck_mini_v2/joystick.py](playground/open_duck_mini_v2/joystick.py)
7. [playground/open_duck_mini_v2/mujoco_infer.py](playground/open_duck_mini_v2/mujoco_infer.py)

这样可以比较快建立对项目全貌的理解。

---

## 14. 参考信息

- 项目 README 提到该仓库受到 `mujoco_playground` 启发；
- imitation reward 的参考运动可由外部仓库生成；
- 当前核心机器人目录为 `playground/open_duck_mini_v2/`。

如果后续需要，也可以继续补充一版：

- 更偏“开发者上手”的 INSTRUCTION.md
- 更偏“训练参数说明”的文档
- 更偏“模块接口说明”的技术文档
