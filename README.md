# applied-genai-api

FastAPI app for Applied Generative AI. It includes the class activity's text generator, Assignment 1 word embeddings, and Assignment 2 image classification.

## Run with Docker

```bash
git clone https://github.com/hiharryyim/applied-genai-api.git
cd applied-genai-api
docker build -t applied-genai-api:assignment2 .
docker run --rm -p 8001:80 applied-genai-api:assignment2
```

Open [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) to try the endpoints.

The first build downloads the dependencies, including the spaCy model. The trained CNN weights are included in the repository. The container runs inference on CPU.

## Classify an image

From the repository directory, run:

```bash
curl -X POST http://127.0.0.1:8001/classify \
  -F "file=@examples/test_image.png"
```

For the included CIFAR-10 test image, the model predicts `cat`. The response contains `class_id`, `class_name`, and `confidence`:

```json
{
  "class_id": 3,
  "class_name": "cat",
  "confidence": 0.7061259746551514
}
```

You can also upload a PNG or JPEG through `/docs`. Images are converted to RGB, resized to 64 x 64, and converted to tensors, using the same preprocessing as training. Unreadable image files return HTTP 400.

## Assignment 2 model

The PyTorch CNN follows the assignment architecture: two 3 x 3 convolutional layers with 16 and 32 filters, each followed by ReLU and 2 x 2 max pooling. The output is flattened to 8,192 features, followed by a 100-unit layer, ReLU, and a 10-unit output layer.

I trained it on [CIFAR-10 from Hugging Face](https://huggingface.co/datasets/uoft-cs/cifar10), using Adam with a learning rate of 0.001 for 10 epochs. Test accuracy was **65.82%** (6,582 of 10,000 images).

The checkpoint is saved in `app/models/cifar10_cnn.pth`. It contains the weights, class names, input size, and training metadata. The API loads it once at startup.

The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Predictions are limited to these classes.

The model definition and training code are in [notebooks/Assignment2.ipynb](notebooks/Assignment2.ipynb). To run it, install `torch`, `torchvision`, `datasets`, `matplotlib`, and `tqdm` in the notebook's kernel environment.

## Run locally

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run fastapi dev app/main.py --port 8000
```

Local API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | `{"Hello": "World"}` |
| POST | `/generate` | `{"start_word": "the", "length": 12}` | generated text |
| POST | `/embedding` | `{"word": "apple"}` | 300-dim vector for the word |
| POST | `/similarity` | `{"word1": "apple", "word2": "orange"}` | cosine similarity |
| POST | `/classify` | image upload (`file`, multipart form) | class ID, name, and confidence |

`/embedding` and `/similarity` return 404 if a word is not in the model's vocabulary.

Example:

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```
