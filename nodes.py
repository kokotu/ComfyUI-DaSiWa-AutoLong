from __future__ import annotations

import copy
import math
import os
import shutil
import subprocess
import threading
import uuid

import folder_paths
import server
import torch


_SESSIONS: dict[str, dict] = {}
_LOCK = threading.RLock()


def _close_process(session: dict, terminate: bool = False) -> None:
    process = session.get("process")
    if process is None:
        return

    try:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    if terminate and process.poll() is None:
        process.terminate()

    try:
        process.wait(timeout=15 if terminate else 300)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)
    finally:
        session["process"] = None


def _reset_session(session_id: str) -> dict:
    old = _SESSIONS.get(session_id)
    if old:
        _close_process(old, terminate=True)

    session = {
        "run_id": uuid.uuid4().hex,
        "process": None,
        "last_frame": None,
        "rewind_frame": None,
        "pending_tail": None,
        "last_latent": None,
        "output_path": None,
        "subfolder": "",
        "filename": "",
        "width": None,
        "height": None,
        "frame_rate": None,
    }
    _SESSIONS[session_id] = session
    return session


def _find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/conda/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("FFmpeg not found. Install FFmpeg or ComfyUI-VideoHelperSuite first.")


def _next_output_path(filename_prefix: str) -> tuple[str, str, str]:
    output_dir = folder_paths.get_output_directory()
    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir
    )
    os.makedirs(full_folder, exist_ok=True)
    output_name = f"{filename}_{counter:05}.mp4"
    return os.path.join(full_folder, output_name), subfolder, output_name


def _start_encoder(
    session: dict,
    width: int,
    height: int,
    frame_rate: float,
    crf: int,
    preset: str,
    filename_prefix: str,
) -> None:
    output_path, subfolder, filename = _next_output_path(filename_prefix)
    command = [
        _find_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(frame_rate)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )
    session.update(
        process=process,
        output_path=output_path,
        subfolder=subfolder,
        filename=filename,
        width=width,
        height=height,
        frame_rate=float(frame_rate),
    )


def _write_frames(session: dict, frames, skip_count: int) -> int:
    process = session["process"]
    start = max(0, int(skip_count))
    if start >= int(frames.shape[0]):
        raise RuntimeError(
            f"Cannot skip {start} frames from a segment containing only "
            f"{int(frames.shape[0])} frames."
        )
    written = 0

    try:
        for frame in frames[start:]:
            array = (
                frame.detach()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .byte()
                .cpu()
                .contiguous()
                .numpy()
            )
            process.stdin.write(array[..., :3].tobytes())
            written += 1
    except (BrokenPipeError, OSError) as error:
        details = b""
        if process.stderr:
            details = process.stderr.read()
        message = details.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg stopped while encoding: {message or error}") from error

    return written


def _frames_to_uint8_cpu(frames):
    return (
        frames.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .cpu()
        .contiguous()[..., :3]
        .clone()
    )


def _write_uint8_frames(session: dict, frames) -> int:
    process = session["process"]
    written = 0
    try:
        for frame in frames:
            process.stdin.write(frame[..., :3].contiguous().numpy().tobytes())
            written += 1
    except (BrokenPipeError, OSError) as error:
        details = b""
        if process.stderr:
            details = process.stderr.read()
        message = details.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg stopped while encoding: {message or error}") from error
    return written


def _write_smart_cut_frames(session: dict, previous_frames, current_frames):
    """Choose a late low-difference cut; never alpha-blend two generated poses."""
    count = int(previous_frames.shape[0])
    if count != int(current_frames.shape[0]):
        raise RuntimeError(
            "Warm-up overlap size changed between segments: "
            f"{count} previous frames vs {int(current_frames.shape[0])} current frames."
        )
    if tuple(previous_frames.shape[1:3]) != tuple(current_frames.shape[1:3]):
        raise RuntimeError("Warm-up overlap resolution changed between segments.")

    # Compare small CPU previews.  The search is restricted to the latter half
    # of the overlap so the new generation has already had time to start moving.
    height, width = int(previous_frames.shape[1]), int(previous_frames.shape[2])
    preview_step = max(1, min(height, width) // 96)
    previous_preview = previous_frames[
        :, ::preview_step, ::preview_step, :3
    ].float().div(255.0)
    current_preview = (
        current_frames[:, ::preview_step, ::preview_step, :3]
        .detach()
        .clamp(0.0, 1.0)
        .cpu()
        .float()
    )

    first_candidate = max(2, math.ceil((count - 1) * 0.55))
    last_candidate = min(count - 2, math.floor((count - 1) * 0.90))
    if first_candidate > last_candidate:
        cut_position = max(1, min(count - 1, count // 2))
        best_score = 0.0
    else:
        target_position = (count - 1) * 0.75
        cut_position = first_candidate
        best_score = float("inf")
        for candidate in range(first_candidate, last_candidate + 1):
            boundary_jump = (
                previous_preview[candidate - 1] - current_preview[candidate]
            ).abs().mean().item()
            same_time_difference = (
                previous_preview[candidate] - current_preview[candidate]
            ).abs().mean().item()
            previous_motion = (
                previous_preview[candidate - 1]
                - previous_preview[candidate - 2]
            ).abs().mean().item()
            current_motion = (
                current_preview[candidate + 1] - current_preview[candidate]
            ).abs().mean().item()
            motion_difference = abs(previous_motion - current_motion)
            timing_penalty = abs(candidate - target_position) / count
            score = (
                boundary_jump
                + 0.25 * same_time_difference
                + 0.15 * motion_difference
                + 0.02 * timing_penalty
            )
            if score < best_score:
                best_score = score
                cut_position = candidate

    written = _write_uint8_frames(session, previous_frames[:cut_position])
    written += _write_frames(
        session, current_frames[cut_position:], skip_count=0
    )
    return written, cut_position, best_score


def _finish_encoder(session: dict) -> None:
    process = session["process"]
    try:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        details = process.stderr.read() if process.stderr else b""
        return_code = process.wait(timeout=300)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait(timeout=15)
        raise RuntimeError("FFmpeg timed out while finalizing the video.") from error
    finally:
        session["process"] = None

    if return_code != 0:
        message = details.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed with code {return_code}: {message}")


def _requeue_current_workflow(session_id: str, next_iteration: int) -> None:
    prompt_queue = server.PromptServer.instance.prompt_queue
    if len(prompt_queue.currently_running) != 1:
        raise RuntimeError("AutoLong requires exactly one currently running ComfyUI prompt.")

    value = next(iter(prompt_queue.currently_running.values()))
    if len(value) == 6:
        _, _, prompt, extra_data, outputs_to_execute, sensitive = value
    else:
        _, _, prompt, extra_data, outputs_to_execute = value
        sensitive = {}

    prompt = copy.deepcopy(prompt)
    updated = False
    for node_id, node in prompt.items():
        if (
            node.get("class_type") == "DaSiWaAutoLongStart"
            and str(node_id) == str(session_id)
        ):
            node.setdefault("inputs", {})["iteration"] = int(next_iteration)
            updated = True
            break

    if not updated:
        raise RuntimeError("Could not find the matching AutoLong Start node in the prompt.")

    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(uuid.uuid4())
    prompt_queue.put(
        (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
    )


class DaSiWaAutoLongStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_image": ("IMAGE",),
                "total_segments": (
                    "INT",
                    {"default": 3, "min": 1, "max": 100, "step": 1},
                ),
                "iteration": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "LATENT", "INT")
    RETURN_NAMES = (
        "start_image",
        "index",
        "total",
        "session_id",
        "previous_latent",
        "motion_latent_count",
    )
    FUNCTION = "begin_segment"
    CATEGORY = "DaSiWa/Auto Long Video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def begin_segment(self, initial_image, total_segments, iteration, unique_id):
        session_id = str(unique_id)
        iteration = int(iteration)
        total_segments = int(total_segments)

        with _LOCK:
            if iteration == 0:
                _reset_session(session_id)
                start_image = initial_image
                previous_latent = {"samples": torch.zeros((1, 16, 1, 1, 1))}
                motion_latent_count = 0
            else:
                session = _SESSIONS.get(session_id)
                continuation_frame = None
                if session:
                    continuation_frame = session.get("rewind_frame")
                    if continuation_frame is None:
                        continuation_frame = session.get("last_frame")
                if continuation_frame is None:
                    raise RuntimeError(
                        "AutoLong continuation state is missing. Set iteration back to 0 and queue again."
                    )
                start_image = continuation_frame
                if session.get("last_latent") is None:
                    previous_latent = {"samples": torch.zeros((1, 16, 1, 1, 1))}
                    motion_latent_count = 0
                else:
                    previous_latent = session["last_latent"]
                    motion_latent_count = 1

        print(f"[DaSiWa AutoLong] Segment {iteration + 1}/{total_segments}")
        return (
            start_image,
            iteration,
            total_segments,
            session_id,
            previous_latent,
            motion_latent_count,
        )


class DaSiWaAutoLongStreamSVIPro:
    PRESETS = ["ultrafast", "veryfast", "fast", "medium", "slow"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "index": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "session_id": ("STRING", {"forceInput": True}),
                "frame_rate": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 240.0}),
                "overlap_frames": (
                    "INT",
                    {"default": 1, "min": 1, "max": 32, "step": 1},
                ),
                "interpolation_multiplier": (
                    "INT",
                    {"default": 2, "min": 1, "max": 16, "step": 1},
                ),
                "crf": ("INT", {"default": 17, "min": 0, "max": 51, "step": 1}),
                "preset": (cls.PRESETS, {"default": "medium"}),
                "filename_prefix": (
                    "STRING",
                    {"default": "video/DaSiWa_AUTO_LONG"},
                ),
            },
            "optional": {
                "samples": ("LATENT",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES", "STRING")
    RETURN_NAMES = ("filenames", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "append_segment"
    CATEGORY = "DaSiWa/Auto Long Video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def append_segment(
        self,
        frames,
        last_frame,
        index,
        total_segments,
        session_id,
        frame_rate,
        overlap_frames,
        interpolation_multiplier,
        crf,
        preset,
        filename_prefix,
        prompt=None,
        unique_id=None,
        samples=None,
    ):
        index = int(index)
        total_segments = int(total_segments)
        overlap_frames = int(overlap_frames)
        interpolation_multiplier = int(interpolation_multiplier)
        if index < 0 or index >= total_segments:
            raise RuntimeError(f"Invalid segment index {index} for total {total_segments}.")
        if len(frames.shape) != 4 or frames.shape[-1] < 3:
            raise RuntimeError(f"Expected IMAGE frames in NHWC format, got {tuple(frames.shape)}.")

        height, width = int(frames.shape[1]), int(frames.shape[2])
        if width % 2 or height % 2:
            raise RuntimeError("H.264 yuv420p output requires even width and height.")

        with _LOCK:
            session = _SESSIONS.get(str(session_id))
            if session is None:
                raise RuntimeError("AutoLong session not found. Set iteration to 0 and queue again.")

            if index == 0:
                _start_encoder(
                    session,
                    width,
                    height,
                    float(frame_rate),
                    int(crf),
                    preset,
                    filename_prefix,
                )
            elif session.get("process") is None:
                raise RuntimeError("AutoLong FFmpeg stream is not active.")

            if (session["width"], session["height"]) != (width, height):
                raise RuntimeError("Frame size changed between segments; continuous encoding is impossible.")
            if abs(session["frame_rate"] - float(frame_rate)) > 0.001:
                raise RuntimeError("Frame rate changed between segments.")

            # A continued SVI clip begins with the previous clip's motion frames.
            # After Nx interpolation, those duplicated frames occupy
            # multiplier * (overlap - 1) + 1 output frames.  Skip exactly that
            # prefix while retaining the newly interpolated transition from the
            # final overlap frame into the first new frame.
            skip_count = 0
            if index > 0:
                skip_count = interpolation_multiplier * (overlap_frames - 1) + 1

            written = _write_frames(session, frames, skip_count=skip_count)
            if int(last_frame.shape[0]) < overlap_frames:
                raise RuntimeError(
                    f"Need {overlap_frames} raw continuation frames, but received "
                    f"only {int(last_frame.shape[0])}."
                )
            session["last_frame"] = (
                last_frame[-overlap_frames:].detach().cpu().clone()
            )
            if samples is not None:
                if not isinstance(samples, dict) or "samples" not in samples:
                    raise RuntimeError("Expected a LATENT dictionary containing 'samples'.")
                session["last_latent"] = {
                    "samples": samples["samples"].detach().cpu().clone()
                }
            output_path = session["output_path"]

            is_final = index + 1 >= total_segments
            if is_final:
                _finish_encoder(session)
            else:
                _requeue_current_workflow(str(session_id), index + 1)

        print(
            f"[DaSiWa AutoLong] Wrote {written} frames for segment "
            f"{index + 1}/{total_segments}: {output_path}"
        )

        filenames = (True, [output_path])
        if is_final:
            ui = {
                "gifs": [
                    {
                        "filename": session["filename"],
                        "subfolder": session["subfolder"],
                        "type": "output",
                        "format": "video/h264-mp4",
                    }
                ],
                "text": [f"Completed {total_segments} segments: {output_path}"],
            }
        else:
            ui = {
                "text": [
                    f"Segment {index + 1}/{total_segments} complete; next segment queued automatically."
                ]
            }
        return {"ui": ui, "result": (filenames, output_path)}


class DaSiWaAutoLongStream(DaSiWaAutoLongStreamSVIPro):
    """Backward-compatible single-tail-frame writer used by existing workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "index": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "session_id": ("STRING", {"forceInput": True}),
                "frame_rate": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 240.0}),
                "crf": ("INT", {"default": 17, "min": 0, "max": 51, "step": 1}),
                "preset": (cls.PRESETS, {"default": "medium"}),
                "filename_prefix": (
                    "STRING",
                    {"default": "video/DaSiWa_AUTO_LONG"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    FUNCTION = "append_segment_legacy"

    def append_segment_legacy(
        self,
        frames,
        last_frame,
        index,
        total_segments,
        session_id,
        frame_rate,
        crf,
        preset,
        filename_prefix,
        prompt=None,
        unique_id=None,
    ):
        return super().append_segment(
            frames=frames,
            last_frame=last_frame,
            index=index,
            total_segments=total_segments,
            session_id=session_id,
            frame_rate=frame_rate,
            overlap_frames=1,
            interpolation_multiplier=2,
            crf=crf,
            preset=preset,
            filename_prefix=filename_prefix,
            prompt=prompt,
            unique_id=unique_id,
            samples=None,
        )


class DaSiWaAutoLongWarmupStream:
    """GGUF-only writer that hides motion restarts inside a short overlap."""

    PRESETS = DaSiWaAutoLongStreamSVIPro.PRESETS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "raw_frames": ("IMAGE",),
                "index": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "session_id": ("STRING", {"forceInput": True}),
                "frame_rate": (
                    "FLOAT",
                    {"default": 32.0, "min": 1.0, "max": 240.0},
                ),
                "warmup_frames": (
                    "INT",
                    {"default": 16, "min": 1, "max": 32, "step": 1},
                ),
                "interpolation_multiplier": (
                    "INT",
                    {"default": 2, "min": 1, "max": 16, "step": 1},
                ),
                "crf": ("INT", {"default": 17, "min": 0, "max": 51, "step": 1}),
                "preset": (cls.PRESETS, {"default": "medium"}),
                "filename_prefix": (
                    "STRING",
                    {"default": "video/DaSiWa_GGUF_WARMUP"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES", "STRING")
    RETURN_NAMES = ("filenames", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "append_segment"
    CATEGORY = "DaSiWa/Auto Long Video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def append_segment(
        self,
        frames,
        raw_frames,
        index,
        total_segments,
        session_id,
        frame_rate,
        warmup_frames,
        interpolation_multiplier,
        crf,
        preset,
        filename_prefix,
        prompt=None,
        unique_id=None,
    ):
        index = int(index)
        total_segments = int(total_segments)
        warmup_frames = int(warmup_frames)
        interpolation_multiplier = int(interpolation_multiplier)
        if index < 0 or index >= total_segments:
            raise RuntimeError(f"Invalid segment index {index} for total {total_segments}.")
        if len(frames.shape) != 4 or frames.shape[-1] < 3:
            raise RuntimeError(f"Expected IMAGE frames in NHWC format, got {tuple(frames.shape)}.")
        if len(raw_frames.shape) != 4 or raw_frames.shape[-1] < 3:
            raise RuntimeError(
                f"Expected raw IMAGE frames in NHWC format, got {tuple(raw_frames.shape)}."
            )

        raw_count = int(raw_frames.shape[0])
        stride_raw = raw_count - warmup_frames - 1
        if stride_raw <= warmup_frames:
            raise RuntimeError(
                "The raw segment is too short for this warm-up. Increase the generated "
                "frame count or reduce warmup_frames."
            )

        expected_frames = interpolation_multiplier * (raw_count - 1) + 1
        actual_frames = int(frames.shape[0])
        if actual_frames != expected_frames:
            raise RuntimeError(
                "Unexpected interpolated frame count: received "
                f"{actual_frames}, expected {expected_frames} from {raw_count} raw frames "
                f"at {interpolation_multiplier}x."
            )

        tail_start = interpolation_multiplier * stride_raw
        tail_count = actual_frames - tail_start
        if tail_start <= tail_count:
            raise RuntimeError(
                "Warm-up overlap is longer than the visible segment. Reduce warmup_frames."
            )

        height, width = int(frames.shape[1]), int(frames.shape[2])
        if width % 2 or height % 2:
            raise RuntimeError("H.264 yuv420p output requires even width and height.")

        with _LOCK:
            session = _SESSIONS.get(str(session_id))
            if session is None:
                raise RuntimeError("AutoLong session not found. Set iteration to 0 and queue again.")

            if index == 0:
                _start_encoder(
                    session,
                    width,
                    height,
                    float(frame_rate),
                    int(crf),
                    preset,
                    filename_prefix,
                )
            elif session.get("process") is None:
                raise RuntimeError("AutoLong FFmpeg stream is not active.")

            if (session["width"], session["height"]) != (width, height):
                raise RuntimeError("Frame size changed between segments; continuous encoding is impossible.")
            if abs(session["frame_rate"] - float(frame_rate)) > 0.001:
                raise RuntimeError("Frame rate changed between segments.")

            if index == 0:
                written = _write_frames(session, frames[:tail_start], skip_count=0)
            else:
                pending_tail = session.get("pending_tail")
                if pending_tail is None:
                    raise RuntimeError(
                        "Warm-up overlap state is missing. Set iteration back to 0 and queue again."
                    )
                if int(pending_tail.shape[0]) != tail_count:
                    raise RuntimeError(
                        "Warm-up settings changed between segments. Keep frame count, "
                        "warmup_frames and interpolation_multiplier unchanged."
                    )
                written, cut_position, cut_score = _write_smart_cut_frames(
                    session, pending_tail, frames[:tail_count]
                )
                written += _write_frames(
                    session, frames[tail_count:tail_start], skip_count=0
                )
                print(
                    "[DaSiWa AutoLong Smart Cut] "
                    f"Overlap cut at {cut_position}/{tail_count - 1}, "
                    f"score={cut_score:.5f}"
                )

            output_path = session["output_path"]
            is_final = index + 1 >= total_segments
            if is_final:
                session["pending_tail"] = None
                session["rewind_frame"] = None
                _finish_encoder(session)
            else:
                rewind_frame = raw_frames[stride_raw : stride_raw + 1]
                session["rewind_frame"] = rewind_frame.detach().cpu().clone()
                session["last_frame"] = session["rewind_frame"]
                session["pending_tail"] = _frames_to_uint8_cpu(
                    frames[tail_start:]
                )
                _requeue_current_workflow(str(session_id), index + 1)

        print(
            f"[DaSiWa AutoLong Warm-up] Wrote {written} frames for segment "
            f"{index + 1}/{total_segments}: {output_path}"
        )

        filenames = (True, [output_path])
        if is_final:
            ui = {
                "gifs": [
                    {
                        "filename": session["filename"],
                        "subfolder": session["subfolder"],
                        "type": "output",
                        "format": "video/h264-mp4",
                    }
                ],
                "text": [
                    f"Completed {total_segments} GGUF warm-up segments: {output_path}"
                ],
            }
        else:
            ui = {
                "text": [
                    f"Segment {index + 1}/{total_segments} complete; "
                    "next warm-up segment queued automatically."
                ]
            }
        return {"ui": ui, "result": (filenames, output_path)}


NODE_CLASS_MAPPINGS = {
    "DaSiWaAutoLongStart": DaSiWaAutoLongStart,
    "DaSiWaAutoLongStream": DaSiWaAutoLongStream,
    "DaSiWaAutoLongStreamSVIPro": DaSiWaAutoLongStreamSVIPro,
    "DaSiWaAutoLongWarmupStream": DaSiWaAutoLongWarmupStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWaAutoLongStart": "DaSiWa AutoLong Start",
    "DaSiWaAutoLongStream": "DaSiWa AutoLong Stream Writer",
    "DaSiWaAutoLongStreamSVIPro": "DaSiWa AutoLong SVI Pro Stream Writer",
    "DaSiWaAutoLongWarmupStream": "DaSiWa AutoLong GGUF Warm-up Stream Writer",
}
