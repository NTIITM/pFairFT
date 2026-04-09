#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
激活值缓存相关类
用于存储和管理模型激活值
"""

import os
import numpy as np
import torch


class DiskActivationCache:
    """
    基于磁盘的激活值缓存类（注意力头级别）。
    
    使用 numpy.memmap 实现内存映射，避免将大量数据加载到内存中。
    形状为 [Num_Samples, Num_Layers, Num_Heads, Head_Dim]。
    """

    def __init__(self, num_samples, num_layers, num_heads, head_dim, name_prefix, temp_dir):
        """
        初始化磁盘激活值缓存。
        
        Args:
            num_samples: 样本数量
            num_layers: 层数
            num_heads: 每层的注意力头数量
            head_dim: 每个头的维度
            name_prefix: 缓存文件的前缀
            temp_dir: 临时目录路径
        """
        # Shape: [N, L, H, D]
        self.shape = (num_samples, num_layers, num_heads, head_dim)
        self.filename = os.path.join(temp_dir, f"{name_prefix}_activations.dat")
        self.data = np.memmap(self.filename, dtype="float32", mode="w+", shape=self.shape)

    def save_batch(self, batch_indices, layer_idx, activations):
        """
        保存一批样本在指定层的激活值。
        
        Args:
            batch_indices: 批次索引（numpy array 或 list）
            layer_idx: 层索引
            activations: 激活值数组，形状为 [batch, num_heads, head_dim]
        """
        # activations: [batch, num_heads, head_dim]
        self.data[batch_indices, layer_idx, :, :] = activations

    def get_batch_layer(self, batch_indices, layer_idx):
        """
        读取特定 Batch 在特定 Layer 的所有头数据。
        
        Args:
            batch_indices: 批次索引（可以是 torch.Tensor 或 numpy array）
            layer_idx: 层索引
            
        Returns:
            激活值数组，形状为 [Batch, Num_Heads, Head_Dim]
        """
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.numpy()
        return self.data[batch_indices, layer_idx, :, :]
    
    def get_column(self, layer_idx, head_idx):
        """
        读取特定 Layer、特定 Head 的所有样本数据。
        
        Args:
            layer_idx: 层索引
            head_idx: 头索引
            
        Returns:
            激活值数组，形状为 [Num_Samples, Head_Dim]
        """
        return self.data[:, layer_idx, head_idx, :]


class MLPDiskCache:
    """
    基于磁盘的 MLP 激活缓存类。

    用于存储每个样本、每一层在最后一个 token 位置的 MLP 输出向量：
    形状为 [Num_Samples, Num_Layers, Hidden_Size]。
    """

    def __init__(self, num_samples: int, num_layers: int, hidden_size: int, name_prefix: str, temp_dir: str):
        """
        初始化 MLP 磁盘缓存。

        Args:
            num_samples: 样本数 N
            num_layers: 层数 L
            hidden_size: 隐藏维度大小
            name_prefix: 缓存文件名前缀（例如 "cf" 或 "fact"）
            temp_dir: 临时目录路径
        """
        self.shape = (num_samples, num_layers, hidden_size)
        self.filename = os.path.join(temp_dir, f"{name_prefix}_mlp_acts.dat")
        self.data = np.memmap(self.filename, dtype="float32", mode="w+", shape=self.shape)

    def save_batch(self, batch_indices, layer_idx: int, activations: np.ndarray):
        """
        保存一批样本在指定层的 MLP 激活值。

        Args:
            batch_indices: 批次索引（numpy array 或 list）
            layer_idx: 层索引
            activations: 激活值数组，形状为 [batch_size, hidden_size]
        """
        self.data[batch_indices, layer_idx, :] = activations

    def get_batch_layer(self, batch_indices, layer_idx: int):
        """
        读取特定 Batch 在特定 Layer 的 MLP 激活值。

        Args:
            batch_indices: 批次索引（可以是 torch.Tensor 或 numpy array）
            layer_idx: 层索引

        Returns:
            激活值数组，形状为 [Batch, Hidden_Size]
        """
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.numpy()
        return self.data[batch_indices, layer_idx, :]
