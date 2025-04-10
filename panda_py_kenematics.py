import torch
import math


def fk(q):
    """
    Forward kinematics implementation in PyTorch.

    Args:
        q: Joint angles tensor of shape (batch_size, 7) or (7,)

    Returns:
        pose: Homogeneous transformation matrix of shape (batch_size, 4, 4) or (4, 4)
    """
    # Handle both batched and non-batched inputs
    input_is_batched = True
    if q.dim() == 1:
        q = q.unsqueeze(0)  # Add batch dimension
        input_is_batched = False

    batch_size = q.shape[0]

    # Create output tensor
    pose = torch.zeros(batch_size, 4, 4, device=q.device, dtype=q.dtype)

    # Precompute sin and cos values for each joint angle
    sin_q = torch.sin(q)
    cos_q = torch.cos(q)

    # Convenience shorthand for sin and cos of each joint
    s0, s1, s2, s3, s4, s5, s6 = sin_q[:, 0], sin_q[:, 1], sin_q[:, 2], sin_q[:, 3], sin_q[:, 4], sin_q[:, 5], sin_q[:,
                                                                                                               6]
    c0, c1, c2, c3, c4, c5, c6 = cos_q[:, 0], cos_q[:, 1], cos_q[:, 2], cos_q[:, 3], cos_q[:, 4], cos_q[:, 5], cos_q[:,
                                                                                                               6]

    # For the term sin(q[6] + M_PI_4) and cos(q[6] + M_PI_4) which appear frequently
    M_PI_4 = math.pi / 4
    s6_pi4 = torch.sin(q[:, 6] + M_PI_4)
    c6_pi4 = torch.cos(q[:, 6] + M_PI_4)

    # Populate the pose matrix element by element following the C++ implementation

    # Row 0, Column 0
    pose[:, 0, 0] = (
            1.0 * s0 * s2 * s3 * s5 * s6_pi4 +
            1.0 * s0 * s2 * s4 * c3 * c6_pi4 -
            1.0 * s0 * s2 * s6_pi4 * c3 * c4 * c5 -
            1.0 * s0 * s4 * s6_pi4 * c2 * c5 -
            1.0 * s0 * c2 * c4 * c6_pi4 -
            1.0 * s1 * s3 * s4 * c0 * c6_pi4 +
            1.0 * s1 * s3 * s6_pi4 * c0 * c4 * c5 +
            1.0 * s1 * s5 * s6_pi4 * c0 * c3 -
            1.0 * s2 * s4 * s6_pi4 * c0 * c1 * c5 -
            1.0 * s2 * c0 * c1 * c4 * c6_pi4 -
            1.0 * s3 * s5 * s6_pi4 * c0 * c1 * c2 -
            1.0 * s4 * c0 * c1 * c2 * c3 * c6_pi4 +
            1.0 * s6_pi4 * c0 * c1 * c2 * c3 * c4 * c5
    )

    # Row 0, Column 1
    pose[:, 0, 1] = (
            1.0 * s0 * s2 * s3 * s5 * c6_pi4 -
            1.0 * s0 * s2 * s4 * s6_pi4 * c3 -
            1.0 * s0 * s2 * c3 * c4 * c5 * c6_pi4 -
            1.0 * s0 * s4 * c2 * c5 * c6_pi4 +
            1.0 * s0 * s6_pi4 * c2 * c4 +
            1.0 * s1 * s3 * s4 * s6_pi4 * c0 +
            1.0 * s1 * s3 * c0 * c4 * c5 * c6_pi4 +
            1.0 * s1 * s5 * c0 * c3 * c6_pi4 -
            1.0 * s2 * s4 * c0 * c1 * c5 * c6_pi4 +
            1.0 * s2 * s6_pi4 * c0 * c1 * c4 -
            1.0 * s3 * s5 * c0 * c1 * c2 * c6_pi4 +
            1.0 * s4 * s6_pi4 * c0 * c1 * c2 * c3 +
            1.0 * c0 * c1 * c2 * c3 * c4 * c5 * c6_pi4
    )

    # Row 0, Column 2
    pose[:, 0, 2] = -1.0 * (
            ((s0 * s2 - c0 * c1 * c2) * c3 - s1 * s3 * c0) * c4 +
            (s0 * c2 + s2 * c0 * c1) * s4
    ) * s5 - 1.0 * (
                            (s0 * s2 - c0 * c1 * c2) * s3 + s1 * c0 * c3
                    ) * c5

    # Row 0, Column 3
    pose[:, 0, 3] = (
            -0.21000000000000002 * (
            ((s0 * s2 - c0 * c1 * c2) * c3 - s1 * s3 * c0) * c4 +
            (s0 * c2 + s2 * c0 * c1) * s4
    ) * s5 -
            0.087999999999999995 * (
                    ((s0 * s2 - c0 * c1 * c2) * c3 - s1 * s3 * c0) * c4 +
                    (s0 * c2 + s2 * c0 * c1) * s4
            ) * c5 +
            0.087999999999999995 * (
                    (s0 * s2 - c0 * c1 * c2) * s3 + s1 * c0 * c3
            ) * s5 -
            0.21000000000000002 * (
                    (s0 * s2 - c0 * c1 * c2) * s3 + s1 * c0 * c3
            ) * c5 +
            0.38400000000000001 * (s0 * s2 - c0 * c1 * c2) * s3 +
            0.082500000000000004 * (s0 * s2 - c0 * c1 * c2) * c3 -
            0.082500000000000004 * s0 * s2 -
            0.082500000000000004 * s1 * s3 * c0 +
            0.38400000000000001 * s1 * c0 * c3 +
            0.316 * s1 * c0 +
            0.082500000000000004 * c0 * c1 * c2
    )

    # Row 1, Column 0
    pose[:, 1, 0] = (
            -1.0 * s0 * s1 * s3 * s4 * c6_pi4 +
            1.0 * s0 * s1 * s3 * s6_pi4 * c4 * c5 +
            1.0 * s0 * s1 * s5 * s6_pi4 * c3 -
            1.0 * s0 * s2 * s4 * s6_pi4 * c1 * c5 -
            1.0 * s0 * s2 * c1 * c4 * c6_pi4 -
            1.0 * s0 * s3 * s5 * s6_pi4 * c1 * c2 -
            1.0 * s0 * s4 * c1 * c2 * c3 * c6_pi4 +
            1.0 * s0 * s6_pi4 * c1 * c2 * c3 * c4 * c5 -
            1.0 * s2 * s3 * s5 * s6_pi4 * c0 -
            1.0 * s2 * s4 * c0 * c3 * c6_pi4 +
            1.0 * s2 * s6_pi4 * c0 * c3 * c4 * c5 +
            1.0 * s4 * s6_pi4 * c0 * c2 * c5 +
            1.0 * c0 * c2 * c4 * c6_pi4
    )

    # Row 1, Column 1
    pose[:, 1, 1] = (
            1.0 * s0 * s1 * s3 * s4 * s6_pi4 +
            1.0 * s0 * s1 * s3 * c4 * c5 * c6_pi4 +
            1.0 * s0 * s1 * s5 * c3 * c6_pi4 -
            1.0 * s0 * s2 * s4 * c1 * c5 * c6_pi4 +
            1.0 * s0 * s2 * s6_pi4 * c1 * c4 -
            1.0 * s0 * s3 * s5 * c1 * c2 * c6_pi4 +
            1.0 * s0 * s4 * s6_pi4 * c1 * c2 * c3 +
            1.0 * s0 * c1 * c2 * c3 * c4 * c5 * c6_pi4 -
            1.0 * s2 * s3 * s5 * c0 * c6_pi4 +
            1.0 * s2 * s4 * s6_pi4 * c0 * c3 +
            1.0 * s2 * c0 * c3 * c4 * c5 * c6_pi4 +
            1.0 * s4 * c0 * c2 * c5 * c6_pi4 -
            1.0 * s6_pi4 * c0 * c2 * c4
    )

    # Row 1, Column 2
    pose[:, 1, 2] = 1.0 * (
            ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 -
            (s0 * s2 * c1 - c0 * c2) * s4
    ) * s5 + 1.0 * (
                            (s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3
                    ) * c5

    # Row 1, Column 3
    pose[:, 1, 3] = (
            0.21000000000000002 * (
            ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 -
            (s0 * s2 * c1 - c0 * c2) * s4
    ) * s5 +
            0.087999999999999995 * (
                    ((s0 * c1 * c2 + s2 * c0) * c3 + s0 * s1 * s3) * c4 -
                    (s0 * s2 * c1 - c0 * c2) * s4
            ) * c5 -
            0.087999999999999995 * (
                    (s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3
            ) * s5 +
            0.21000000000000002 * (
                    (s0 * c1 * c2 + s2 * c0) * s3 - s0 * s1 * c3
            ) * c5 -
            0.38400000000000001 * (s0 * c1 * c2 + s2 * c0) * s3 -
            0.082500000000000004 * (s0 * c1 * c2 + s2 * c0) * c3 -
            0.082500000000000004 * s0 * s1 * s3 +
            0.38400000000000001 * s0 * s1 * c3 +
            0.316 * s0 * s1 +
            0.082500000000000004 * s0 * c1 * c2 +
            0.082500000000000004 * s2 * c0
    )

    # Row 2, Column 0
    pose[:, 2, 0] = (
            1.0 * s1 * s2 * s4 * s6_pi4 * c5 +
            1.0 * s1 * s2 * c4 * c6_pi4 +
            1.0 * s1 * s3 * s5 * s6_pi4 * c2 +
            1.0 * s1 * s4 * c2 * c3 * c6_pi4 -
            1.0 * s1 * s6_pi4 * c2 * c3 * c4 * c5 -
            1.0 * s3 * s4 * c1 * c6_pi4 +
            1.0 * s3 * s6_pi4 * c1 * c4 * c5 +
            1.0 * s5 * s6_pi4 * c1 * c3
    )

    # Row 2, Column 1
    pose[:, 2, 1] = (
            1.0 * s1 * s2 * s4 * c5 * c6_pi4 -
            1.0 * s1 * s2 * s6_pi4 * c4 +
            1.0 * s1 * s3 * s5 * c2 * c6_pi4 -
            1.0 * s1 * s4 * s6_pi4 * c2 * c3 -
            1.0 * s1 * c2 * c3 * c4 * c5 * c6_pi4 +
            1.0 * s3 * s4 * s6_pi4 * c1 +
            1.0 * s3 * c1 * c4 * c5 * c6_pi4 +
            1.0 * s5 * c1 * c3 * c6_pi4
    )

    # Row 2, Column 2
    pose[:, 2, 2] = -1.0 * (
            (s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4
    ) * s5 - 1.0 * (
                            s1 * s3 * c2 + c1 * c3
                    ) * c5

    # Row 2, Column 3
    pose[:, 2, 3] = (
            -0.21000000000000002 * (
            (s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4
    ) * s5 -
            0.087999999999999995 * (
                    (s1 * c2 * c3 - s3 * c1) * c4 - s1 * s2 * s4
            ) * c5 +
            0.087999999999999995 * (
                    s1 * s3 * c2 + c1 * c3
            ) * s5 -
            0.21000000000000002 * (
                    s1 * s3 * c2 + c1 * c3
            ) * c5 +
            0.38400000000000001 * s1 * s3 * c2 +
            0.082500000000000004 * s1 * c2 * c3 -
            0.082500000000000004 * s1 * c2 -
            0.082500000000000004 * s3 * c1 +
            0.38400000000000001 * c1 * c3 +
            0.316 * c1 + 0.33300000000000002
    )

    # Row 3, Column 3
    pose[:, 3, 3] = 1.0

    # Return without batch dimension if input wasn't batched
    if not input_is_batched:
        pose = pose.squeeze(0)

    return torch.cat([pose[0:3, 0], pose[0:3, 1], pose[0:3, 2], pose[0:3, 3]])
if __name__ =="__main__":
    q = torch.tensor([0.5853594 ,  2.3372703, - 0.65501136,  1.0937663,   0.48222056,  1.4050351,
     - 0.92837393])
    t = fk(q)
    print(t)