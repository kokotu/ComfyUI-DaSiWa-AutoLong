# ComfyUI-DaSiWa-AutoLong

为 ComfyUI 视频工作流提供一次 Queue 的长视频自动续写节点。

## 工作方式

- 一个 ComfyUI 工作流，一次点击 Queue。
- 每次只生成一段，上一段尾帧自动作为下一段首帧。
- 每段结束后自动重新排队，因此上一段的大批图像可以释放。
- 已插帧的帧直接写入同一个 FFmpeg 编码流，不经过 MP4 分段拼接。
- 第二段开始自动跳过重复的第一帧，避免接缝停顿。
- 不依赖 Easy-Use 循环节点，不在 RAM 中累计完整视频。

## RunPod 安装

在 Jupyter Terminal 中执行：

```bash
cd /workspace/runpod-slim/ComfyUI/custom_nodes
git clone https://github.com/kokotu/ComfyUI-DaSiWa-AutoLong.git
```

已经安装时更新：

```bash
git -C /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-DaSiWa-AutoLong pull
```

随后重启 ComfyUI。

本节点没有额外 Python requirements，但系统必须有 `ffmpeg`。安装了
ComfyUI-VideoHelperSuite 的常规 RunPod 环境通常已经具备 FFmpeg。

## 节点

- `DaSiWa AutoLong Start`
  - 输入初始首帧和总段数。
  - 后续任务自动输出上一段保存的尾帧。
  - `iteration` 必须保持为 0；自动续写时由节点内部更新。
- `DaSiWa AutoLong Stream Writer`
  - 接收当前段的完整帧、当前段尾帧、段号和帧率。
  - 第一段启动 FFmpeg，后续段写入同一个视频编码流。
  - 第二段开始自动跳过重复首帧。
  - 最后一段完成时关闭文件并返回最终 MP4。

节点仓库只包含节点代码，不包含任何第三方工作流或工作流 JSON。

输出位于 ComfyUI 的 `output/video` 目录，默认文件名前缀为
`DaSiWa_AUTO_LONG`。

## 注意事项

- 自动任务完成前不要再次 Queue 同一工作流。
- 如果手动中断任务，再次运行前确保 `iteration` 为 0。
- 所有段的分辨率和帧率必须一致。
- 编码器使用 H.264 `libx264`、`yuv420p`，默认 CRF 17。
