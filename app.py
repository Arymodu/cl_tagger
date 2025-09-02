import gradio as gr
import numpy as np
from PIL import Image
import json
import os
import io
import requests
import matplotlib.pyplot as plt
import matplotlib
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time


# import spaces
# 创建一个虚拟的 spaces 模块用于本地运行
class DummySpaces:
    def GPU(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


spaces = DummySpaces()

import onnxruntime as ort
import torch
import timm
from safetensors.torch import load_file as safe_load_file
import shutil
import zipfile
from datetime import datetime
import glob


# --- Data Classes and Helper Functions ---
@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    artist: list[np.int64]
    character: list[np.int64]
    copyright: list[np.int64]
    meta: list[np.int64]
    quality: list[np.int64]
    model: list[np.int64]


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height: return image
    new_size = max(width, height)
    new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def load_tag_mapping(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        tag_mapping_data = json.load(f)
    if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
        idx_to_tag = {int(k): v for k, v in tag_mapping_data["idx_to_tag"].items()}
        tag_to_category = tag_mapping_data["tag_to_category"]
    elif isinstance(tag_mapping_data, dict):
        try:
            tag_mapping_data_int_keys = {int(k): v for k, v in tag_mapping_data.items()}
            idx_to_tag = {idx: data['tag'] for idx, data in tag_mapping_data_int_keys.items()}
            tag_to_category = {data['tag']: data['category'] for data in tag_mapping_data_int_keys.values()}
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Unsupported tag mapping format (dict): {e}. Expected int keys with 'tag' and 'category'.")
    else:
        raise ValueError("Unsupported tag mapping format: Expected a dictionary.")

    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating, general, artist, character, copyright, meta, quality, model_name = [], [], [], [], [], [], [], []
    for idx, tag in idx_to_tag.items():
        if idx >= len(names):
            names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        category = tag_to_category.get(tag, 'Unknown')
        idx_int = int(idx)
        if category == 'Rating':
            rating.append(idx_int)
        elif category == 'General':
            general.append(idx_int)
        elif category == 'Artist':
            artist.append(idx_int)
        elif category == 'Character':
            character.append(idx_int)
        elif category == 'Copyright':
            copyright.append(idx_int)
        elif category == 'Meta':
            meta.append(idx_int)
        elif category == 'Quality':
            quality.append(idx_int)
        elif category == 'Model':
            model_name.append(idx_int)

    # 添加类别信息
    category_info = {
        "General": {"indices": general, "count": len(general), "default": True},
        "Character": {"indices": character, "count": len(character), "default": True},
        "Copyright": {"indices": copyright, "count": len(copyright), "default": True},
        "Artist": {"indices": artist, "count": len(artist), "default": False},
        "Meta": {"indices": meta, "count": len(meta), "default": False},
        "Quality": {"indices": quality, "count": len(quality), "default": False},
        "Model": {"indices": model_name, "count": len(model_name), "default": False},
        "Rating": {"indices": rating, "count": len(rating), "default": False}
    }

    return LabelData(names=names, rating=np.array(rating, dtype=np.int64), general=np.array(general, dtype=np.int64),
                     artist=np.array(artist, dtype=np.int64), character=np.array(character, dtype=np.int64),
                     copyright=np.array(copyright, dtype=np.int64), meta=np.array(meta, dtype=np.int64),
                     quality=np.array(quality, dtype=np.int64),
                     model=np.array(model_name, dtype=np.int64)), category_info


def preprocess_image(image: Image.Image, target_size=(448, 448)):
    image = pil_ensure_rgb(image)
    image = pil_pad_square(image)
    image_resized = image.resize(target_size, Image.BICUBIC)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = img_array[::-1, :, :]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return image, img_array


def get_tags(probs, labels: LabelData, gen_threshold, char_threshold, enabled_categories):
    result = {"rating": [], "general": [], "character": [], "copyright": [], "artist": [], "meta": [], "quality": [],
              "model": []}

    # 只处理启用的类别
    if "Rating" in enabled_categories and len(labels.rating) > 0:
        valid_indices = labels.rating[labels.rating < len(probs)]
        if len(valid_indices) > 0:
            rating_probs = probs[valid_indices]
            if len(rating_probs) > 0:
                rating_idx_local = np.argmax(rating_probs)
                rating_idx_global = valid_indices[rating_idx_local]
                if rating_idx_global < len(labels.names):
                    result["rating"].append((labels.names[rating_idx_global], float(rating_probs[rating_idx_local])))

    if "Quality" in enabled_categories and len(labels.quality) > 0:
        valid_indices = labels.quality[labels.quality < len(probs)]
        if len(valid_indices) > 0:
            quality_probs = probs[valid_indices]
            if len(quality_probs) > 0:
                quality_idx_local = np.argmax(quality_probs)
                quality_idx_global = valid_indices[quality_idx_local]
                if quality_idx_global < len(labels.names):
                    result["quality"].append(
                        (labels.names[quality_idx_global], float(quality_probs[quality_idx_local])))

    category_map = {
        "General": (labels.general, gen_threshold),
        "Character": (labels.character, char_threshold),
        "Copyright": (labels.copyright, char_threshold),
        "Artist": (labels.artist, char_threshold),
        "Meta": (labels.meta, gen_threshold),
        "Model": (labels.model, gen_threshold)
    }

    for category, (indices, threshold) in category_map.items():
        if category in enabled_categories and len(indices) > 0:
            valid_indices = indices[indices < len(probs)]
            if len(valid_indices) > 0:
                category_probs = probs[valid_indices]
                mask = category_probs >= threshold
                selected_indices_global = valid_indices[np.where(mask)[0]]
                for idx_global in selected_indices_global:
                    if idx_global < len(labels.names):
                        result[category.lower()].append((labels.names[idx_global], float(probs[idx_global])))

    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)
    return result


def format_tags_as_string(predictions: Dict, enabled_categories: List[str]) -> str:
    output_tags = []

    # 将启用的类别转换为小写以匹配predictions的键
    enabled_lower = [cat.lower() for cat in enabled_categories]

    if "rating" in enabled_lower and predictions.get("rating"):
        output_tags.append(predictions["rating"][0][0].replace("_", " "))

    if "quality" in enabled_lower and predictions.get("quality"):
        output_tags.append(predictions["quality"][0][0].replace("_", " "))

    # 映射类别名称
    category_mapping = {
        "artist": "artist",
        "character": "character",
        "copyright": "copyright",
        "general": "general",
        "meta": "meta",
        "model": "model"
    }

    for display_cat, internal_cat in category_mapping.items():
        if display_cat.capitalize() in enabled_categories and predictions.get(internal_cat):
            for tag, prob in predictions[internal_cat]:
                if display_cat == "meta" and any(p in tag.lower() for p in ['id', 'commentary', 'request', 'mismatch']):
                    continue
                output_tags.append(tag.replace("_", " "))

    return ", ".join(output_tags)


def visualize_predictions(predictions: Dict, threshold: float, enabled_categories: List[str]):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(8, 12), dpi=100)
    ax_tags = fig.add_subplot(1, 1, 1)

    all_tags, all_probs, all_colors = [], [], []
    color_map = {
        'rating': 'red',
        'character': 'blue',
        'copyright': 'purple',
        'artist': 'orange',
        'general': 'green',
        'meta': 'gray',
        'quality': 'yellow',
        'model': 'cyan'
    }

    # 只显示启用的类别
    for cat, color in color_map.items():
        # 将类别名称转换为首字母大写的格式
        display_cat = cat.capitalize()
        if display_cat in enabled_categories:
            for tag, prob in predictions.get(cat, []):
                all_tags.append(f"[{cat[0].upper()}] {tag.replace('_', ' ')}")
                all_probs.append(prob)
                all_colors.append(color)

    if not all_tags:
        ax_tags.text(0.5, 0.5, "No tags found", ha='center', va='center')
    else:
        y_pos = np.arange(len(all_tags))
        ax_tags.barh(y_pos, all_probs, color=all_colors)
        ax_tags.set_yticks(y_pos)
        ax_tags.set_yticklabels(all_tags)
        ax_tags.invert_yaxis()
        ax_tags.set_xlabel('Probability')
        ax_tags.set_title(f'Tags (Threshold ≳ {threshold:.2f})')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# --- Constants & Globals ---
REPO_ID = "cella110n/cl_tagger"
MODEL_OPTIONS = {
    "cl_tagger_1_00": "cl_tagger_1_00/model_optimized.onnx",
    "cl_tagger_1_01": "cl_tagger_1_01/model_optimized.onnx"
}
DEFAULT_MODEL = "cl_tagger_1_01"
CACHE_DIR = os.path.abspath("./model_cache")
OUTPUT_DIR = os.path.abspath("./batch_output")

g_onnx_model_path, g_tag_mapping_path, g_labels_data, g_category_info, g_current_model = None, None, None, None, None


# --- Core Logic ---
def initialize_app(model_choice=DEFAULT_MODEL):
    global g_onnx_model_path, g_tag_mapping_path, g_labels_data, g_category_info, g_current_model
    if g_current_model == model_choice and g_labels_data is not None:
        return f"Model {model_choice} is already loaded."

    g_current_model = model_choice
    onnx_filename = MODEL_OPTIONS[model_choice]
    tag_mapping_filename = f"{model_choice}/tag_mapping.json"

    try:
        hf_token = os.environ.get("HF_TOKEN")
        g_onnx_model_path = hf_hub_download(repo_id=REPO_ID, filename=onnx_filename, cache_dir=CACHE_DIR,
                                            token=hf_token)
        g_tag_mapping_path = hf_hub_download(repo_id=REPO_ID, filename=tag_mapping_filename, cache_dir=CACHE_DIR,
                                             token=hf_token)
        g_labels_data, g_category_info = load_tag_mapping(g_tag_mapping_path)
        status_msg = f"Successfully initialized and loaded model: {model_choice}"
    except Exception as e:
        g_current_model = None  # Reset on failure
        raise gr.Error(f"Initialization failed: {e}. Check logs and HF_TOKEN.")

    # 在初始化后添加检查 - 在return语句之前
    try:
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            print(f"模型 {model_choice} 已加载，并使用 GPU (CUDA) 进行推理。")
        else:
            print(f"模型 {model_choice} 已加载，但未检测到 GPU，使用 CPU 进行推理。")
    except:
        pass

    return status_msg


def get_onnx_session():
    """
    创建并返回ONNX Runtime推理会话，优先使用GPU。
    """
    try:
        # 获取可用的执行提供程序
        available_providers = ort.get_available_providers()

        # 优先使用 CUDA (GPU)，如果可用
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("使用 GPU (CUDA) 进行推理。")
        else:
            providers = ['CPUExecutionProvider']
            print("未检测到 GPU，使用 CPU 进行推理。")

        # 创建会话选项（可选，用于一些配置）
        sess_options = ort.SessionOptions()
        # 例如，设置线程数
        # sess_options.intra_op_num_threads = 4
        # sess_options.inter_op_num_threads = 4

        # 创建 InferenceSession
        session = ort.InferenceSession(g_onnx_model_path, sess_options=sess_options, providers=providers)
        return session

    except Exception as e:
        print(f"创建 ONNX 会话时出错: {e}")
        # 如果创建失败，尝试回退到 CPU
        try:
            print("尝试回退到 CPU...")
            session = ort.InferenceSession(g_onnx_model_path, providers=['CPUExecutionProvider'])
            return session
        except Exception as e2:
            raise gr.Error(f"无法创建 ONNX 会话 (GPU 和 CPU 均失败): {e2}")


def run_inference(session, image_pil, gen_threshold, char_threshold, enabled_categories):
    _, input_tensor = preprocess_image(image_pil)
    input_tensor = input_tensor.astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})[0]

    if np.isnan(outputs).any() or np.isinf(outputs).any():
        outputs = np.nan_to_num(outputs)

    probs = 1 / (1 + np.exp(-outputs[0]))  # Sigmoid

    predictions = get_tags(probs, g_labels_data, gen_threshold, char_threshold, enabled_categories)
    tags_string = format_tags_as_string(predictions, enabled_categories)

    return tags_string, predictions


# --- Gradio Interface Functions ---
@spaces.GPU()
def predict_single_image(image_input, model_choice, gen_threshold, char_threshold, output_mode, enabled_categories):
    if image_input is None:
        raise gr.Error("Please upload an image.")

    initialize_app(model_choice)
    session = get_onnx_session()  # 获取会话

    try:
        tags_string, predictions = run_inference(session, image_input, gen_threshold, char_threshold,
                                                 enabled_categories)

        viz_image = None
        if output_mode == "Tags + Visualization":
            viz_image = visualize_predictions(predictions, gen_threshold, enabled_categories)

        return tags_string, viz_image
    finally:
        # 确保无论是否发生异常，都会尝试清理
        del session  # 删除 session 对象
        import gc
        gc.collect()  # 触发 Python 垃圾回收
        # 如果系统中也使用了 PyTorch，可以尝试清空 PyTorch 的 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@spaces.GPU()
def _run_batch_processing_core(image_paths: List[str], model_choice: str, gen_threshold: float, char_threshold: float,
                               enabled_categories: List[str], progress=gr.Progress(track_tqdm=True)):
    """Core logic for batch processing a list of image file paths."""
    initialize_app(model_choice)
    session = get_onnx_session()  # 获取会话

    all_captions_log = ""
    all_captions_list = []
    processed_count = 0

    try:
        for image_path in progress.tqdm(image_paths, desc="Processing Images"):
            try:
                image_pil = Image.open(image_path)
                original_filename = os.path.basename(image_path)
                filename_no_ext = os.path.splitext(original_filename)[0]
                file_dir = os.path.dirname(image_path)

                tags_string, _ = run_inference(session, image_pil, gen_threshold, char_threshold, enabled_categories)

                # 在原位置生成txt文件
                txt_filepath = os.path.join(file_dir, f"{filename_no_ext}.txt")
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(tags_string)

                processed_count += 1
                all_captions_list.append(f"{image_path}: {tags_string}")
                all_captions_log = "\n\n".join(all_captions_list)

                yield all_captions_log, f"Processed {processed_count}/{len(image_paths)} images"  # Stream UI updates

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                all_captions_list.append(f"Error processing {image_path}: {e}")
                all_captions_log = "\n\n".join(all_captions_list)
                yield all_captions_log, f"Processed {processed_count}/{len(image_paths)} images (Error occurred)"
                continue

        completion_msg = f"Processing completed! Generated {processed_count} .txt files alongside the images."
        yield all_captions_log, completion_msg
    finally:
        # 确保在批处理完成后清理 session
        del session
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def batch_from_directory(input_dir, model_choice, gen_threshold, char_threshold, enabled_categories,
                         progress=gr.Progress(track_tqdm=True)):
    if not input_dir or not os.path.isdir(input_dir):
        raise gr.Error("Please provide a valid, existing directory path.")

    # 递归搜索所有子目录中的图片
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"]
    image_paths = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        raise gr.Error(f"No images found in the specified directory: {input_dir}")

    yield from _run_batch_processing_core(image_paths, model_choice, gen_threshold, char_threshold, enabled_categories,
                                          progress)


# --- Gradio UI ---
css = """.gradio-container { font-family: 'IBM Plex Sans', sans-serif; } footer { display: none !important; } .gr-prose { max-width: 100% !important; }"""

with gr.Blocks(css=css, title="CL EVA02 ONNX Tagger") as demo:
    gr.Markdown("<h1>CL EVA02 ONNX Tagger</h1>")
    gr.Markdown(
        "Fine-tuned from `SmilingWolf/wd-eva02-large-tagger-v3`. Use the tabs below for single or batch image tagging.")

    with gr.Row():
        model_choice = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()), value=DEFAULT_MODEL, label="Model Version")
        gen_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.50, label="General Tag Threshold")
        char_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.70,
                                   label="Character/Artist Tag Threshold")

    # 添加标签类别选择
    with gr.Row():
        enabled_categories = gr.CheckboxGroup(
            choices=["General", "Character", "Copyright", "Artist", "Meta", "Quality", "Model", "Rating"],
            value=["General", "Character", "Copyright"],  # 默认选中这三类
            label="Enabled Tag Categories"
        )

    model_status = gr.Textbox(label="Model Status", interactive=False, visible=False)
    model_choice.change(fn=initialize_app, inputs=[model_choice], outputs=[model_status])

    with gr.Tabs():
        # --- Single Image Tab ---
        with gr.TabItem("Single Image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Input Image")
                    output_mode = gr.Radio(choices=["Tags Only", "Tags + Visualization"], value="Tags + Visualization",
                                           label="Output Mode")
                    predict_button = gr.Button("Predict Single Image", variant="primary")
                with gr.Column(scale=1):
                    output_tags = gr.Textbox(label="Predicted Tags", lines=10, interactive=False)
                    output_visualization = gr.Image(type="pil", label="Prediction Visualization", interactive=False)

        # --- Batch Processing Tab ---
        with gr.TabItem("Batch Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input_dir = gr.Textbox(label="Enter Input Directory Path",
                                                 placeholder="e.g., C:\\Users\\YourUser\\Pictures\\MyDataset")
                    batch_dir_button = gr.Button("Run Batch from Directory", variant="primary")
                    gr.Markdown(
                        "**Note:** This will recursively process all images in subfolders and save .txt files alongside each image.")
                with gr.Column(scale=3):
                    batch_output_captions = gr.Textbox(label="Processing Log", lines=15, interactive=False)
                    batch_completion_msg = gr.Textbox(label="Status", value="Ready", interactive=False)

    # --- Event Handlers ---
    predict_button.click(
        fn=predict_single_image,
        inputs=[image_input, model_choice, gen_threshold, char_threshold, output_mode, enabled_categories],
        outputs=[output_tags, output_visualization]
    )

    batch_dir_button.click(
        fn=batch_from_directory,
        inputs=[batch_input_dir, model_choice, gen_threshold, char_threshold, enabled_categories],
        outputs=[batch_output_captions, batch_completion_msg]
    )

# --- Main Block ---
if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN environment variable not set. Downloads may be rate-limited.")

    initialize_app()
    demo.launch(server_name="0.0.0.0", server_port=7870, share=False)