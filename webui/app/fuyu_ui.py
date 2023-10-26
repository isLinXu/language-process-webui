import gradio as gr
import re
import torch
from PIL import Image
from transformers import AutoTokenizer, FuyuForCausalLM, FuyuImageProcessor, FuyuProcessor

model_id = "adept/fuyu-8b"
dtype = torch.bfloat16
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=dtype)
processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=tokenizer)

CAPTION_PROMPT = "Generate a coco-style caption.\n"
DETAILED_CAPTION_PROMPT = "What is happening in this image?"


def resize_to_max(image, max_width=1920, max_height=1080):
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    width = int(width * scale)
    height = int(height * scale)

    return image.resize((width, height), Image.LANCZOS)


def pad_to_size(image, canvas_width=1920, canvas_height=1080):
    width, height = image.size
    if width >= canvas_width and height >= canvas_height:
        return image

    # Paste at (0, 0)
    canvas = Image.new("RGB", (canvas_width, canvas_height))
    canvas.paste(image)
    return canvas


def predict(image, prompt):
    # image = image.convert('RGB')
    model_inputs = processor(text=prompt, images=[image])
    model_inputs = {k: v.to(dtype=dtype if torch.is_floating_point(v) else v.dtype, device=device) for k, v in
                    model_inputs.items()}

    generation_output = model.generate(**model_inputs, max_new_tokens=50)
    prompt_len = model_inputs["input_ids"].shape[-1]
    return tokenizer.decode(generation_output[0][prompt_len:], skip_special_tokens=True)


def caption(image, detailed_captioning):
    if detailed_captioning:
        caption_prompt = DETAILED_CAPTION_PROMPT
    else:
        caption_prompt = CAPTION_PROMPT
    return predict(image, caption_prompt).lstrip()


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def scale_factor_to_fit(original_size, target_size=(1920, 1080)):
    width, height = original_size
    max_width, max_height = target_size
    if width <= max_width and height <= max_height:
        return 1.0
    return min(max_width / width, max_height / height)


def tokens_to_box(tokens, original_size):
    bbox_start = tokenizer.convert_tokens_to_ids("<0x00>")
    bbox_end = tokenizer.convert_tokens_to_ids("<0x01>")
    try:
        # Assumes a single box
        bbox_start_pos = (tokens == bbox_start).nonzero(as_tuple=True)[0].item()
        bbox_end_pos = (tokens == bbox_end).nonzero(as_tuple=True)[0].item()

        if bbox_end_pos != bbox_start_pos + 5:
            return tokens

        # Retrieve transformed coordinates from tokens
        coords = tokenizer.convert_ids_to_tokens(tokens[bbox_start_pos + 1:bbox_end_pos])

        # Scale back to original image size and multiply by 2
        scale = scale_factor_to_fit(original_size)
        top, left, bottom, right = [2 * int(float(c) / scale) for c in coords]

        # Replace the IDs so they get detokenized right
        replacement = f" <box>{top}, {left}, {bottom}, {right}</box>"
        replacement = tokenizer.tokenize(replacement)[1:]
        replacement = tokenizer.convert_tokens_to_ids(replacement)
        replacement = torch.tensor(replacement).to(tokens)

        tokens = torch.cat([tokens[:bbox_start_pos], replacement, tokens[bbox_end_pos + 1:]], 0)
        return tokens
    except:
        gr.Error("Can't convert tokens.")
        return tokens


def coords_from_response(response):
    # y1, x1, y2, x2
    pattern = r"<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)</box>"

    match = re.search(pattern, response)
    if match:
        # Unpack and change order
        y1, x1, y2, x2 = [int(coord) for coord in match.groups()]
        return (x1, y1, x2, y2)
    else:
        gr.Error("The string is malformed or does not match the expected pattern.")


def localize(image, query):
    prompt = f"When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\n{query}"

    # Downscale and/or pad to 1920x1080
    padded = resize_to_max(image)
    padded = pad_to_size(padded)

    model_inputs = processor(text=prompt, images=[padded])
    model_inputs = {k: v.to(dtype=dtype if torch.is_floating_point(v) else v.dtype, device=device) for k, v in
                    model_inputs.items()}

    generation_output = model.generate(**model_inputs, max_new_tokens=40)
    prompt_len = model_inputs["input_ids"].shape[-1]
    tokens = generation_output[0][prompt_len:]
    tokens = tokens_to_box(tokens, image.size)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    coords = coords_from_response(decoded)
    return image, [(coords, f"Location of \"{query}\"")]


css = """
  #mkd {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
            <h1 id="title">Fuyu Multimodal Demo</h1>
            <h3><a href="https://hf.co/adept/fuyu-8b">Fuyu-8B</a> is a multimodal model that supports a variety of tasks combining text and image prompts.</h3>
            For example, you can use it for captioning by asking it to describe an image. You can also ask it questions about an image, a task known as Visual Question Answering, or VQA. This demo lets you explore captioning and VQA, with more tasks coming soon :)
            Learn more about the model in <a href="https://www.adept.ai/blog/fuyu-8b">our blog post</a>.
            <br>
          	<br>
            <strong>Note: This is a raw model release. We have not added further instruction-tuning, postprocessing or sampling strategies to control for undesirable outputs. The model may hallucinate, and you should expect to have to fine-tune the model for your use-case!</strong>
            <h3>Play with Fuyu-8B in this demo! 💬</h3>
        """
    )
    with gr.Tab("Visual Question Answering"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload your Image", type="pil")
                text_input = gr.Textbox(label="Ask a Question")
            vqa_output = gr.Textbox(label="Output")

        vqa_btn = gr.Button("Answer Visual Question")

        gr.Examples(
            [["assets/vqa_example_1.png", "How is this made?"],
             ["assets/vqa_example_2.png", "What is this flower and where is it's origin?"],
             ["assets/docvqa_example.png", "How many items are sold?"],
             ["assets/screen2words_ui_example.png", "What is this app about?"]],
            inputs=[image_input, text_input],
            outputs=[vqa_output],
            fn=predict,
            cache_examples=True,
            label='Click on any Examples below to get VQA results quickly 👇'
        )

    with gr.Tab("Image Captioning"):
        with gr.Row():
            with gr.Column():
                captioning_input = gr.Image(label="Upload your Image", type="pil")
                detailed_captioning_checkbox = gr.Checkbox(label="Enable detailed captioning")
            captioning_output = gr.Textbox(label="Output")
        captioning_btn = gr.Button("Generate Caption")

        gr.Examples(
            [["assets/captioning_example_1.png", False], ["assets/captioning_example_2.png", True]],
            inputs=[captioning_input, detailed_captioning_checkbox],
            outputs=[captioning_output],
            fn=caption,
            cache_examples=True,
            label='Click on any Examples below to get captioning results quickly 👇'
        )

    captioning_btn.click(fn=caption, inputs=[captioning_input, detailed_captioning_checkbox], outputs=captioning_output)
    vqa_btn.click(fn=predict, inputs=[image_input, text_input], outputs=vqa_output)

    with gr.Tab("Find Text in Screenshots"):
        with gr.Row():
            with gr.Column():
                localization_input = gr.Image(label="Upload your Image", type="pil")
                query_input = gr.Textbox(label="Text to find")
                localization_btn = gr.Button("Locate Text")
            with gr.Column():
                with gr.Row(height=800):
                    localization_output = gr.AnnotatedImage(label="Text Position")

        gr.Examples(
            [["assets/localization_example_1.jpeg", "Share your repair"],
             ["assets/screen2words_ui_example.png", "statistics"]],
            inputs=[localization_input, query_input],
            outputs=[localization_output],
            fn=localize,
            cache_examples=True,
            label='Click on any Examples below to get localization results quickly 👇'
        )

    localization_btn.click(fn=localize, inputs=[localization_input, query_input], outputs=localization_output)

demo.launch(server_name="0.0.0.0")