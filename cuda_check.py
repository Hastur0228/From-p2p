#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA版本检测和PyTorch CUDA可用性检查
CUDA Version Detection and PyTorch CUDA Availability Check
"""

import torch
import subprocess
import sys

def check_cuda_version():
    """检查CUDA版本"""
    try:
        # 使用nvidia-smi命令获取CUDA版本
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # 从输出中提取CUDA版本
            output = result.stdout
            for line in output.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version: ')[1].strip()
                    return cuda_version
        return "CUDA未安装或nvidia-smi不可用"
    except Exception as e:
        return f"检查CUDA版本时出错: {e}"

def check_pytorch_cuda():
    """检查PyTorch CUDA可用性"""
    print("=== PyTorch CUDA 检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # 测试CUDA张量操作
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("CUDA张量运算测试: 成功")
        except Exception as e:
            print(f"CUDA张量运算测试: 失败 - {e}")
    else:
        print("CUDA不可用，PyTorch将使用CPU")

def main():
    """主函数"""
    print("=== CUDA 和 PyTorch 检测工具 ===")
    print()
    
    # 检查系统CUDA版本
    print("=== 系统CUDA版本 ===")
    cuda_version = check_cuda_version()
    print(f"系统CUDA版本: {cuda_version}")
    print()
    
    # 检查PyTorch CUDA
    check_pytorch_cuda()
    print()
    
    # 总结
    print("=== 总结 ===")
    if torch.cuda.is_available():
        print("✅ PyTorch CUDA可用，可以进行GPU加速计算")
    else:
        print("❌ PyTorch CUDA不可用，将使用CPU进行计算")
        print("建议:")
        print("1. 安装CUDA版本的PyTorch")
        print("2. 确保系统已安装CUDA驱动")
        print("3. 检查GPU是否支持CUDA")

if __name__ == "__main__":
    main()
