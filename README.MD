# Panda IK Solver

本项目实现了基于神经网络映射器的小型逆运动学（IK）求解器，针对 Franka Emika Panda 机械臂。通过端到端优化学习输入映射，快速求解给定末端位姿的关节角。

## 功能简介

- **逆运动学求解**  
  给定末端目标位姿（旋转和平移），使用小型 MLP（`panda_Mapper`）回归一组关节角，保证正向运动学输出与目标位姿尽可能接近。

- **正向运动学验证**  
  求解出的关节角通过正向运动学（`panda_forward_kinematics`）还原位姿，并与目标位姿计算位置和旋转误差。

- **鲁棒优化策略**  
  自动检测损失收敛平滑，支持动态重启、学习率调整，提高求解成功率。

## 依赖环境

- Python 3.8+
- PyTorch 1.11+
- NumPy

其他依赖模块（需自行准备）：
- `Mapper.py`：定义 `panda_Mapper` 网络结构
- `panda_py_kenematics.py`：提供 `fk` 正向运动学函数
- `panda_py.py`：提供 C++/Python 加速版的正向运动学接口（可选）

## 文件结构

| 文件名                      | 描述                      |
|:-------------------------|:------------------------|
| `iksolver_panda.py`      | 主程序，包含逆运动学求解器、辅助函数和测试用例 |
| `Mapper.py`              | Panda专用的轻量级神经网络映射器      |
| `panda_py_kenematics.py` | Panda机械臂的正向运动学实现(解析解)   |


## 快速使用

1. 安装依赖：
   ```bash
   pip install torch numpy 
   ```
    note: panda-py的环境不能直接pip 要去git找相应的版本
2. 准备 `Mapper.py`、`panda_kenematics.py` 文件，确保路径正确。

3. 运行逆运动学测试：
   ```bash
   python iksolver_panda.py
   ```

4. 结果示例：
   ```
   运行测试用例 2...
   Iteration 200, Loss: 0.0001234567
   ...
   成功求解, ik为：
   [0.123, -1.234, ...]
   收敛于迭代次数: 1800, 最终损失: 0.00000012

   测试用例 2 验证结果:
   当前位置:
   [...]
   期望位姿:
   [...]
   位置误差: 0.00000567
   旋转误差: 0.00001234
   ```

## 注意事项

- 输入目标位姿必须是 **4×4 齐次变换矩阵**，并转换为特定的 12 维向量格式供网络使用。
- 本IK求解器默认使用**弧度制**，**角度限制**按Panda实际关节范围约束。
- `panda_forward_kinematics` 返回的是打平后的12维向量，需要注意 reshape。
- 本项目适合学习和实验，工程部署需要进一步强化鲁棒性（如加入更多初值、多次重启、误差控制等）。

## 联系作者

- **Author**: Haotian Liang (htLiang)
- **Email**: haotianliang10@gmail.com

## 引用
如果觉得本项目对你有帮助，欢迎引用Xiangjian Li的相关论文：
---
```tex
@ARTICLE{9606543,
  author={Li, Xiangjian and Liu, Huashan and Dong, Menghua},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={A General Framework of Motion Planning for Redundant Robot Manipulator Based on Deep Reinforcement Learning}, 
  year={2022},
  volume={18},
  number={8},
  pages={5253-5263},
  doi={10.1109/TII.2021.3125447},
  ISSN={1941-0050},
  month={Aug},
```

}
