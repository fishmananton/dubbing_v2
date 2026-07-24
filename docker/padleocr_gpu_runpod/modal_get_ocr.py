import modal

app = modal.App("get_ocr")

image = modal.Image.from_registry("docker.io/delascorpion/paddleocr:4")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    startup_timeout=60 * 10,
)
def get_ocr_job(
    input_url: str,
    interval: float = 0.125,
    crop_bottom_fraction: float = 0.33,
    image_quality: int = 2,
    batch_size: int = 20,
    start_time: float = 0.0,
    end_time: float | None = None,
    visual_diff_threshold: float = 2.0,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    lang: str | None = None,
):
    from getocr import (
        download_file,
        extract_frames,
        get_ocr_instance,
        filter_visually_duplicate_frames,
        batch,
        process_batch,
        deduplicate_srts,
    )
    import os
    import glob
    import tempfile
    import shutil

    output_folder = tempfile.mkdtemp(prefix="ocr_video_")
    try:
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        file_path = os.path.join(output_folder, "input.mp4")

        download_file(input_url, file_path)

        params = {
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
        }
        if lang is not None:
            params["lang"] = lang

        extract_frames(
            video_path=file_path,
            output_folder=frames_folder,
            image_quality=image_quality,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            crop_bottom_fraction=crop_bottom_fraction,
        )

        ocr_instance = get_ocr_instance(params)

        img_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
        img_paths = filter_visually_duplicate_frames(
            img_paths,
            diff_threshold=visual_diff_threshold,
        )

        all_results = []
        total_batches = ((len(img_paths) - 1) // batch_size) + 1 if img_paths else 0

        for batch_num, img_batch in enumerate(batch(img_paths, batch_size), start=1):
            print(f"📦 Processing batch {batch_num} from {total_batches}...")
            batch_results = process_batch(
                img_batch,
                ocr_instance,
                interval,
                start_time=start_time,
            )
            all_results.extend(batch_results)

        clear_res = deduplicate_srts(
            srt_list=all_results,
            frame_interval=interval,
        )

        return {"message": "Processing done", "result": clear_res}
    finally:
        shutil.rmtree(output_folder, ignore_errors=True)
