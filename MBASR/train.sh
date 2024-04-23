#!/bin/bash

tag=(1)
# tag=(2)
# tag=(3)
values=(0.2 0.4 0.6 0.8)

# 设置最大并行数
max_parallel=3

# 定义函数来运行训练脚本并重试
run_training() {
  tag=$1
  values=$2

  params="--tag $tag --gamma $values"
  log_file="Res/del/tag_${tag}_values_${values}.txt"
  
  # 尝试运行训练脚本，并将输出重定向到日志文件
  echo "开始训练: tag=${tag}, values=${values}"
  python train.py $params  > $log_file
  
  # 检查训练是否成功
  if [ $? -eq 0 ]; then
    echo "训练成功: tag=${tag}, values=${values}"
  else
    echo "训练失败: tag=${tag}, values=${values}"
    if [ "$interrupted" = true ]; then
      echo "已中断，不再重试"
    else
      echo "重新运行训练"
      # 重新运行训练
      interrupted=true
      run_training $tag $values
    fi
  fi
}

# 初始化并行进程计数器和进程总数
parallel_count=0
total_processes=$(( ${#tag[@]} * ${#values[@]} ))

# 设置中断标志
interrupted=false

# 定义中断处理函数
on_interrupt() {
  echo "接收到中断信号，正在退出..."
  interrupted=true
  exit 1
}

# 捕获中断信号并调用中断处理函数
trap on_interrupt INT

# 遍历参数组合并并行运行
for tag in "${tag[@]}"; do
  for values in "${values[@]}"; do
    # 如果达到最大并行数，等待其中一个进程完成
    if [ $parallel_count -eq $max_parallel ]; then
      wait -n
      parallel_count=$((parallel_count-1))
    fi
      
    # 并行运行训练
    run_training $traintype $tag $values &
    parallel_count=$((parallel_count+1))
      
    # 显示进度信息
    progress=$(( parallel_count * 100 / total_processes ))
    echo "当前进度: $progress%"
    done
  done
done
# 等待剩余的后台进程完成
wait
echo "运行完成"
