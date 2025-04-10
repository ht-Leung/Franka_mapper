# --- coding: utf-8 ---
# @Time    : 11/8/24 6:29 PM
# @Author  : htLiang
# @Email   : ryzeliang@163.com
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Mapper import panda_Mapper
from panda_py_kenematics import fk

def IKsolver(T_Target, max_iterations=5000, convergence_threshold=1e-7, lr=3e-4):
    """
    优化版本的逆运动学求解器

    参数:
        Rot: 目标旋转矩阵（一维数组形式）
        Tras: 目标位置向量
        max_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值
        lr: 学习率

    返回:
        ths_np: 求解的关节角度（弧度制） (7,)
    """
    # 关节角度限制
    angles_limit_tensor = torch.tensor([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    # 初始化映射网络
    mapper = panda_Mapper()

    # 使用Adam优化器
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)

    # 随机输入初始化
    inp = torch.Tensor(np.random.uniform(-1.0, 1.0, 2))


    # 记录损失变化，用于提前终止
    loss_history = []
    plateau_count = 0
    best_loss = float('inf')
    best_ths = None

    for n in range(1, max_iterations + 1):
        # 通过映射网络获取关节角度
        ths = mapper(inp) * angles_limit_tensor

        # 计算正向运动学
        T = fk(ths)

        # 计算损失
        loss = F.mse_loss(T, T_Target)
        # 逐元素计算均方误差
        # element_wise_squared_error = (T - T_Target) ** 2
        # 所有元素的均方误差和
        # loss = torch.mean(element_wise_squared_error)
        # print(tar_T)

        # 每200次迭代打印一次损失
        if n % 200 == 0:
            loss_np = loss.detach().numpy()
            print(f"Iteration {n}, Loss: {loss_np:.10f}")

            # 检查是否有改善
            if len(loss_history) > 0 and abs(loss_np - loss_history[-1]) < 1e-8:
                plateau_count += 1
            else:
                plateau_count = 0

            # 如果连续5次没有改善，尝试重新初始化
            if plateau_count >= 5 and loss_np > 1e-4:
                print("Loss plateaued, reinitializing...")
                inp = torch.Tensor(np.random.uniform(-1.0, 1.0, 2))
                optimizer = torch.optim.Adam(mapper.parameters(), lr=lr * 1.5)
                plateau_count = 0

            loss_history.append(loss_np)

        # 检查是否达到收敛阈值
        if loss < convergence_threshold:
            ths_np = ths.detach().numpy()
            print(f"\n成功求解,ik为：\n{ths_np}")
            print(f"收敛于迭代次数: {n}, 最终损失: {loss.item():.8f}")
            return ths_np

        # 保存最佳结果
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_ths = ths.detach().clone()
            # print(best_ths.shape)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 如果达到最大迭代次数但没有收敛，返回最佳结果
    print(f"\n达到最大迭代次数 ({max_iterations})，返回最佳解")
    print(f"最佳损失: {best_loss:.8f}")
    return best_ths.numpy()


def homogeneous_matrix(R, p):
    """
    构建齐次变换矩阵

    参数:
        R: 3x3旋转矩阵或9元素一维数组
        p: 3元素位置向量

    返回:
        T: 4x4齐次变换矩阵
    """
    T = np.eye(4)
    if R.size == 9:
        T[:3, :3] = R.reshape((3, 3))
    else:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def calculate_pose_error(T_target, T_current):
    """
    计算位姿误差

    参数:
        T_target: 目标齐次变换矩阵
        T_current: 当前齐次变换矩阵

    返回:
        position_error: 位置误差（欧氏距离）
        rotation_error: 旋转误差（Frobenius范数）
    """
    position_error = np.linalg.norm(T_target[:3, 3] - T_current[:3, 3])
    rotation_error = np.linalg.norm(T_target[:3, :3] - T_current[:3, :3])
    return position_error, rotation_error


def test_panda_fk(panda_joint_angles):
    print("Testing corrected panda_FK...")

    # Example joint angles (in radians)
    # panda_joint_angles = torch.tensor([0.0, -np.pi / 4, 0.0, -np.pi / 2, 0.0, np.pi / 3, 0.0])
    # tool_length = 0.212  # Gripper length (meters)

    output = fk(panda_joint_angles)

    print("Panda FK Output (flattened T0_7):")
    print(output)
    print()

    # Shape should be 12 (3 columns for rotation + 1 column for position)
    print(f"Output shape: {output.shape}")

    # Reshape into meaningful components
    position = output[9:]
    rotation_matrix = output[:9].reshape(3, 3)

    print("Position:", position)
    print("Rotation matrix:\n", rotation_matrix)
    print()

if __name__ == "__main__":
    if __name__ == "__main__":
        import panda_py

        print("\n\n运行测试用例 2...")

        # 定义目标齐次变换矩阵
        T_tar_np = np.array([[ 0.25276055,  0.9674985  ,-0.00766289  ,0.32027466],
                            [ 0.9671562 , -0.25243548 , 0.02975191 , 0.49476646],
                            [ 0.02685054 ,-0.01493132 ,-0.99952794 , 0.00360629],
                            [ 0.     ,     0.       ,   0.       ,   1.        ]])

        # 转为Tensor格式
        T_tar_tensor = torch.tensor(T_tar_np, dtype=torch.float32)
        T_tar = torch.cat([
            T_tar_tensor[0:3, 0],
            T_tar_tensor[0:3, 1],
            T_tar_tensor[0:3, 2],
            T_tar_tensor[0:3, 3]
        ])

        # 逆运动学求解
        start_time = time.time()
        ikths = IKsolver(T_tar, convergence_threshold = 1e-12, max_iterations=3000)
        print(f"\n逆解时间：{time.time() - start_time:.4f} 秒")
        print("求得关节角:", ikths.shape)

        # 通过正向运动学验证逆解结果
        T_current = panda_py.fk(ikths)
        position_error, rotation_error = calculate_pose_error(T_tar_np, T_current)

        print("\n测试用例 2 验证结果:")
        print(f"当前位置:\n{T_current}")
        print(f"期望位姿:\n{T_tar_np}")
        print(f"位置误差: {position_error:.8f}")
        print(f"旋转误差: {rotation_error:.8f}")

        # 额外输出一次 fk 结果（可选）
        q = panda_py.fk(ikths)
        # print("\n再次调用 panda_py.fk 的结果:")
        # print(q)

        test_panda_fk(torch.tensor(ikths))
        T_current = panda_py.fk(ikths)
        T_current_adjusted = T_current.copy()
        T_current_adjusted[:3, :3] = T_current[:3, :3].T
        print(T_current_adjusted[:3, :3])

