from PIL import Image
from pathlib import Path
from typing import List, Dict

import os
import io
import base64
import argparse
import numpy as np


def read_json(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data

def read_jsonl(jsonl_file):
    with open(jsonl_file, "r") as f:
        lines = [json.loads(line.rstrip()) for line in f.readlines()]
    return lines

def read_lines(txt_file):
    with open(txt_file, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def write_json(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f)

def write_jsonl(lines, save_path):
    with open(save_path, "w") as f:
        for line in lines:
            json_record = json.dumps(line)
            f.write(json_record + '\n')

def image_to_base64(image_path):
    # Open the image file
    with Image.open(image_path) as image:
        # Convert the image to RGB format if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Save the image into a bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        # Encode the bytes as base64
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')


# Function to generate an HTML file with visualization results given the LLaVA-format annotation
def generate_html_file(
    data_dicts: List[Dict],
    img_root: str,
    save_path: str,
    idxs: List[int] = None,
):
    """
    Inputs
        data_dicts: List of data dicts. Each data is a dictionary that contains three keys
            id: annotation id
            image: relative path to the image
            conversations: conversations between human and gpt
        img_root: root path to the images
        save_path: path to save the output html file
        idxs: List of selected images to visualize. If not given, visualize first 10
    """
    num_data = len(data_dicts)

    if idxs is None:
        idxs = list(np.arange(min(10, num_data)))

    print(f"Visualize {len(idxs)} from {num_data} images.")

    with open(save_path, 'w') as html_file:
        html_file.write('<html><head><title>Visualization Results</title></head><body>')

        # Add header row
        html_file.write('<div style="display: flex; font-weight: bold;">')
        html_file.write('<div style="flex: 1;">ID</div>')
        html_file.write('<div style="flex: 1;">Image</div>')
        html_file.write('<div style="flex: 1;">Conversations</div>')
        html_file.write('</div><br>')

        for idx in idxs:
            if idx >=  num_data:
                raise Warning(f"Invalid index: {idx}")
                continue
            data_dict = data_dicts[idx]
            anno_id = data_dict['id']
            image_path = Path(img_root) / data_dict['image']
            image_in_base64 = image_to_base64(image_path)
            conversations = data_dict['conversations']
            conversation_str = ""
            for conv in conversations:
                conversation_str += f"from: {conv['from']}<br>"
                conversation_str += f"value: {conv['value']}<br>"

            # Create a row for each quadruplet with four columns
            html_file.write('<div style="display: flex;">')

            # Display ID in the first column
            html_file.write(f'<div style="flex: 1;">{anno_id}</div>')

            # Display image in the second column
            # html_file.write(f'<div style="flex: 1;"><img src="{image_path}" alt="Image" width="300"></div>')
            html_file.write(f'<div style="flex: 1;"><img src="data:image/jpeg;base64,{image_in_base64}" alt="Embedded Image" width="300"></div>')

            # Display question in the third column
            html_file.write(f'<div style="flex: 1;">{conversation_str}</div>')

            html_file.write('</div><br>')

        html_file.write('</body></html>')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HTML visualization file from JSONL data.')
    parser.add_argument('--json-file', '--json_file', help='Path to the JSONL file')
    parser.add_argument('--img-root', '--img_root', default='./playground/data', help='Path to the image root directory')
    parser.add_argument('--output-html-file', '--output_html_file', help='Path for the output HTML file')
    parser.add_argument('--output-dir', '--output_dir', default="./vis_html/", help='Path to dir to save output')
    parser.add_argument('--idxs', type=int, nargs='+', default=None, help='List of indices to visualize')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    json_data = read_json(args.json_file)
    generate_html_file(
        json_data,
        args.img_root,
        Path(args.output_dir) / args.output_html_file,
        args.idxs
    )

# PYTHONPATH=./ python tools/vis/gen_webpage_llava.py --json-file /net/acadia14a/data/yumin/exps/LLaVA/human_data/llava_action.json --output-html-file tmp.html --idxs 1 100 1000 10000 15000
