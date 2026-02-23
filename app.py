#!/usr/bin/env python3
"""
Halftone Video Tool — Web UI
Run with: python app.py
Then open: http://localhost:7860
"""

import gradio as gr
import tempfile
import os
from pathlib import Path

from halftone_video import process_video


def run_halftone(video_path, mode, cell_size, angle, invert, max_frames_enabled, max_frames):
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    limit = int(max_frames) if max_frames_enabled else None

    try:
        process_video(
            input_path=Path(video_path),
            output_path=Path(out_path),
            mode=mode,
            cell_size=int(cell_size),
            angle=float(angle),
            invert=invert,
            max_frames=limit,
        )
    except Exception as e:
        os.unlink(out_path)
        raise gr.Error(f"Processing failed: {e}")

    return out_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue="neutral",
    secondary_hue="neutral",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
    .container { max-width: 1100px; margin: auto; }
    #title { text-align: center; margin-bottom: 8px; }
    #subtitle { text-align: center; color: #888; margin-bottom: 24px; font-size: 14px; }
    .process-btn { min-height: 48px; font-size: 16px !important; }
    footer { display: none !important; }
"""

with gr.Blocks(title="Halftone Video Tool") as demo:

    with gr.Column(elem_classes="container"):
        gr.Markdown("# Halftone Video Tool", elem_id="title")
        gr.Markdown(
            "Upload a video and convert it to a halftone dot-pattern style — "
            "classic **B&W** or **CMYK colour** screen-printing simulation.",
            elem_id="subtitle",
        )

        with gr.Row(equal_height=False):
            # ---- Left column: controls ----
            with gr.Column(scale=1, min_width=280):
                video_in = gr.Video(label="Input Video", sources=["upload"])

                mode = gr.Radio(
                    choices=["bw", "color"],
                    value="bw",
                    label="Mode",
                    info="bw = grayscale dots on white  |  color = CMYK simulation",
                )

                cell_size = gr.Slider(
                    minimum=4, maximum=40, value=10, step=1,
                    label="Cell Size (px)",
                    info="Controls dot density — smaller = finer detail, larger = bolder dots",
                )

                angle = gr.Slider(
                    minimum=0, maximum=175, value=45, step=5,
                    label="Grid Angle (°)",
                    info="Rotation of the halftone grid (B&W mode only)",
                )

                invert = gr.Checkbox(
                    label="Invert dots",
                    info="Bright areas → big dots, dark areas → small dots",
                )

                with gr.Accordion("Performance options", open=False):
                    limit_toggle = gr.Checkbox(
                        label="Limit frames (for quick preview)",
                        value=False,
                    )
                    limit_frames = gr.Slider(
                        minimum=10, maximum=300, value=60, step=10,
                        label="Max frames to process",
                        visible=False,
                    )
                    limit_toggle.change(
                        fn=lambda v: gr.update(visible=v),
                        inputs=limit_toggle,
                        outputs=limit_frames,
                    )

                gr.Markdown(
                    "> **Tip:** Colour mode is ~4× slower than B&W. "
                    "Use frame limiting for a quick preview while dialling in settings.",
                    elem_id="tip",
                )

                process_btn = gr.Button(
                    "Process Video", variant="primary", elem_classes="process-btn"
                )

            # ---- Right column: output ----
            with gr.Column(scale=1, min_width=320):
                video_out = gr.Video(label="Output Video", interactive=False)

        process_btn.click(
            fn=run_halftone,
            inputs=[video_in, mode, cell_size, angle, invert, limit_toggle, limit_frames],
            outputs=video_out,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=THEME,
        css=CSS,
    )
