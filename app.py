from flask import Flask, render_template, request
import os

from PIL import Image
import torch
from model import *


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = f.filename
        f.save(saveLocation)
        image = Image.open(saveLocation)
        im = transform(image).unsqueeze(0)
        print(im.size)
        outputs = model(im)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
        os.remove(saveLocation)
        name = plot_results(image, probas[keep], bboxes_scaled)

        # respond with the inference
        return render_template('inference.html', name = name)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
