# ComfyUI-DaSiWa-AutoLong

为 ComfyUI 视频工作流提供一次 Queue 的长视频自动续写节点。

## 工作方式

- 一个 ComfyUI 工作流，一次点击 Queue。
- 每次只生成一段，可将上一段最后 1 帧或多帧自动作为下一段的动作上下文。
- 每段结束后自动重新排队，因此上一段的大批图像可以释放。
- 已插帧的帧直接写入同一个 FFmpeg 编码流，不经过 MP4 分段拼接。
- 支持多帧重叠；会按插帧倍数精确跳过重复区，保留从最后一帧重叠帧到新内容的插值过渡。
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
  - 后续任务自动输出上一段保存的尾部动作帧。
  - `iteration` 必须保持为 0；自动续写时由节点内部更新。
- `DaSiWa AutoLong Stream Writer`
  - 保留给旧工作流使用，按单尾帧重叠处理，原有 JSON 不会因更新节点而失效。
- `DaSiWa AutoLong SVI Pro Stream Writer`
  - 接收当前段的完整帧、当前段原始帧、段号和帧率。
  - 第一段启动 FFmpeg，后续段写入同一个视频编码流。
  - `overlap_frames` 是传给下一段的原始动作帧数；普通尾帧续写设为 1，SVI 2.0 设为 5，SVI 2.0 Pro 的一个时间 latent 对应 4 帧，设为 4。
  - `interpolation_multiplier` 必须与前面的插帧倍数一致；不插帧设为 1，16→32 FPS 设为 2。
  - 第二段开始自动跳过完整的重复区。
  - 最后一段完成时关闭文件并返回最终 MP4。
  - 最终完成后在同一个节点内显示播放器和下载按钮。

节点仓库只包含节点代码，不包含任何第三方工作流或工作流 JSON。

输出位于 ComfyUI 的 `output/video` 目录，默认文件名前缀为
`DaSiWa_AUTO_LONG`。

## 注意事项

- 自动任务完成前不要再次 Queue 同一工作流。
- 如果手动中断任务，再次运行前确保 `iteration` 为 0。
- 所有段的分辨率和帧率必须一致。
- SVI 2.0 工作流必须保持 `overlap_frames = 5`；SVI 2.0 Pro 必须保持 `overlap_frames = 4`。16→32 FPS 时两者的 `interpolation_multiplier` 都设为 2。
- 编码器使用 H.264 `libx264`、`yuv420p`，默认 CRF 17。
