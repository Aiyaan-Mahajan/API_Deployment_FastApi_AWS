# Deploying Machine Learning Models with FastAPI and Docker on AWS

## Introduction

Many blog posts and tutorials focus on the training of machine learning and deep learning models. However, there is not a lot of information about how to create and deploy these models so they can be easily used as part of a website or another system.

This blog post describes approaches for creating web APIs for these models, and their deployment to AWS.

Although the focus of this post is on [TensorFlow 2](https://www.tensorflow.org/), with a few modifications the code can also support other frameworks such as [PyTorch](https://pytorch.org/) or [LightGBM](https://lightgbm.readthedocs.io/en/latest/).

## About Me

My name is Adam and I develop machine learning / deep learning systems and prototypes at the AI Technology R&D Division of Proto Solution.

Most of my work focuses on the development of deep learning models using TensorFlow, and using these to develop APIs or batch applications that can be deployed onto Amazon Web Services (AWS).

## Approaches

There are a number of AWS services that can be used to deploy machine learning models, for example;

- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [AWS Lambda](https://aws.amazon.com/lambda/)
- [AWS Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/)
- [AWS Batch](https://aws.amazon.com/batch/)
- [AWS Elastic Container Service (ECS)](https://aws.amazon.com/ecs/)

Each of these services vary in cost, complexity and features.

For this blog we will focus on deploying to **AWS Lambda** and **AWS Elastic Beanstalk**; I find these services the easiest to use and most of my prototypes and applications are deployed onto them.

I hope to focus on other services in a future blog.

## API

An, API &mdash; Application Programming Interface &mdash; describes how a user or another system can communicate with the application without having to know its internal details.

A web API allows an application to expose its functionality over the internet using HTTP or other protocols.

In this tutorial we will create a web API using the FastAPI framework and use Docker and AWS services to deploy it. 

We will use a Tensorflow [MobileNetV2](https://arxiv.org/abs/1801.04381) model that has been trained on the [ImageNet](https://www.image-net.org/) dataset as a example.

## FastAPI

[FastAPI](https://fastapi.tiangolo.com/) is a web-framework for building web APIs in Python.

Although other frameworks, such as [Flask](https://flask.palletsprojects.com/en/2.0.x/), are also very popular I believe that FastAPI has a number of advantages over these, in particular:

- It is really fast and lightweight.
- Builtin data serialization and validation with [Pydantic](https://pydantic-docs.helpmanual.io/).
- Builtin Open API documentation generation.

## Docker

[Docker](https://www.docker.com/) has become the defacto way to deploy applications. 

It allows developers to package the application and all its dependencies, including the operating system, into what is known as a Docker image.

These images can be run on a variety of different platforms that support the Docker engine. 

Images that are run on the Docker engine are known as *containers*.

Docker provides a number of different commands for managing containers, but for this tutorial we will be just be using the [build](https://docs.docker.com/engine/reference/commandline/build/) and [run](https://docs.docker.com/engine/reference/commandline/run/) commands.

## Tutorial

In this section we will introduce and describe each part of the API. The full code will be available at the end of this section.

### Install Dependencies

Our first step is to create a new virtual environment &mdash;an isolated, working copy of Python. 

For this tutorial, I will use [Conda](https://docs.conda.io/en/latest/) to create a new Python 3.7 virtual environment and activate it.

```bash
conda create -n ApiDeployment python=3.7
conda activate ApiDeployment
```

We can then install the dependencies, listed in the `requirements.txt` file into the virtual environment:

```bash
pip install -r requirements.txt
```

Let's have a look at some of the dependent packages that we will be using:

```bash
Pillow==8.2.0
tensorflow==2.4.2
numpy==1.19.5
fastapi==0.65.2
pydantic==1.8.2
aiohttp==3.7.3
uvloop==0.14.0
uvicorn[standard]==0.14.0
gunicorn==20.1.0
aiofiles==0.7.0
mangum==0.12.2
```

### Imports and Logging

The first thing we need to do is import the libraries that the API will use.

I will explain their usage as we go through the code.

```python
import argparse
import base64
import io
import os
import logging
import sys

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

from urllib.parse import urlparse

from aiohttp.client import ClientSession
from asyncio import wait_for, gather, Semaphore

from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel, validator

import numpy as np

from PIL import Image

from mangum import Mangum
```

The API will be configured using environment variables; these can be easily passed to Docker on the command line.

```python
THREAD_COUNT = int(os.environ.get('THREAD_COUNT', 5))
"""The number of threads used to download and process image content."""

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
"""The number of images to process in a batch."""

TIMEOUT = int(os.environ.get('TIMEOUT', 30))
"""The timeout to use when downloading files."""
```

We will also create a `logging` object to write log messages. 

```python
logger = logging.getLogger(__name__)
```

### Data Model

Our next step is to create the data structures that define the interface to our API by using the [Pydantic](https://pydantic-docs.helpmanual.io/) package.

Detailed tutorials for creating the data structures can be found [here](https://pydantic-docs.helpmanual.io/usage/models/). 

```python
class HealthCheck(BaseModel):
    """
    Represents an image to be predicted.
    """
    message: Optional[str] = 'OK'


class ImageInput(BaseModel):
    """
    Represents an image to be predicted.
    """
    url: Optional[str] = None
    data: Optional[str] = None


class ImageOutput(BaseModel):
    """
    Represents the result of a prediction
    """
    score: Optional[float] = 0.0
    category: Optional[str] = None
    name: Optional[str] = None

    @validator('score')
    def result_check(cls, v):
        return round(v, 4)


class PredictRequest(BaseModel):
    """
    Represents a request to process
    """
    images: List[ImageInput] = []


class PredictResponse(BaseModel):
    """
    Represents a request to process
    """
    images: List[ImageOutput] = []
```

The `PredictRequest` object represents the data that is passed into our API; that is, the images that we want to process.

The `PredictResponse` object defines how the prediction results are returned back to the caller of the API.

By defining these structures, we can automatically deserialize and validate [JSON](https://www.json.org/) requests containing image URLs or Base64 encoded image data. For example:

```json
{
  "images": [
    {
      "url": "https://localhost/test.jpg"
    }
  ]
}
```

We can also serialize a JSON response containing the ImageNet category prediction, and confidence score for each image. For example:

```json
{
    "images": [
        {
            "score": 0.508,
            "category": "n03770679",
            "name": "minivan"
        }
    ]
}
```

### FastAPI Application

The next step is to instantiate the FastAPI application object; this allows us to use a number of annotations described in the next steps.

```python
app = FastAPI()
```

### Exception Handlers

The following exception handlers are used generate error messages, that will be returned to the caller, if an error occurs during processing. 

```python
class ImageNotDownloadedException(Exception):
    pass


@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for all other errors.
    """
    return JSONResponse(status_code=500, content={'message': 'Internal error.'})


@app.exception_handler(ImageNotDownloadedException)
async def client_exception_handler(request: Request, exc: ImageNotDownloadedException):
    """
    Called when the image could not be downloaded.
    """
    return JSONResponse(status_code=400, content={'message': 'One or more images could not be downloaded.'})
```

### Image Classifier

The following class defines and downloads a pretrained MobileNetV2 model.

By calling the `predict` function we can use the model to predict the ImageNet categories for a list of images. 

Images are pre-processed and resized to fit the input of the model (224 pixels x 224 pixels).

```python
class ImageClassifier:
    """
    Classifies images according to ImageNet categories.
    """
    def __init__(self):
        """
        Prepares the model used by the application for use.
        """
        self.model = MobileNetV2()
        _, height, width, channels = self.model.input_shape
        self.input_width = width
        self.input_height = height
        self.input_channels = channels

    def _prepare_images(self, images):
        """
        Prepares the images for prediction.

        :param images: The list of images to prepare for prediction in Pillow Image format.

        :return: A list of processed images.
        """
        batch = np.zeros((len(images), self.input_height, self.input_width, self.input_channels), dtype=np.float32)
        for i, image in enumerate(images):
            x = image.resize((self.input_width, self.input_height), Image.BILINEAR)
            batch[i, :] = np.array(x, dtype=np.float32)
        batch = preprocess_input(batch)
        return batch

    def predict(self, images, batch_size):
        """
        Predicts the category of each image.

        :param images: A list of images to classify.
        :param batch_size: The number of images to process at once.

        :return: A list containing the predicted category and confidence score for each image.
        """
        batch = self._prepare_images(images)
        scores = self.model.predict(batch, batch_size)
        results = decode_predictions(scores, top=1)
        return results
```

### Configure Logging

Next, we will create a function to configure the logger for the application.

All messages will be output to `stdout`. This will allow messages to be easily logged by AWS Lambda and Elastic Beanstalk later on.
 
```python
def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the application.
    """
    root = logging.getLogger()
    root.handlers.clear()
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    root.setLevel(logging_level)
    root.addHandler(stream_handler)
```


### Load Model

This step will create and load the `ImageClassifier` object when the application is started.

It will also configure the logging for the application using the function defined in the previous section.

```python
@app.on_event('startup')
def load_model():
    """
    Loads the model prior to the first request.
    """
    configure_logging()
    logger.info('Loading model...')
    app.state.model = ImageClassifier()
```

We will store the `ImageClassifier` object in the application `state`, but it is also possible to store the model as a global variable instead.

### Image Processing

In this section we define a number of functions to download images from a URL or decode Base64 image data stored in the request message.

We will use the `aiohttp` package to perform the download and convert the data into `Pillow` images. 

```python
def get_url_scheme(url, default_scheme='unknown'):
    """
    Returns the scheme of the specified URL or 'unknown' if it could not be determined.
    """
    result = urlparse(url, scheme=default_scheme)
    return result.scheme


async def retrieve_content(entry, sess, sem):
    """
    Retrieves the image content for the specified entry.
    """
    raw_data = None
    if entry.data is not None:
        raw_data = base64.b64decode(entry.data)
    elif entry.url is not None:
        source_uri = entry.url
        scheme = get_url_scheme(source_uri)
        if scheme in ('http', 'https'):
            raw_data = await download(source_uri, sess, sem)
        else:
            raise ValueError('Invalid scheme: %s' % scheme)
    if raw_data is not None:
        image = Image.open(io.BytesIO(raw_data))
        image = image.convert('RGB')
        return image
    return None


async def retrieve_images(entries):
    """
    Retrieves the images for processing.

    :param entries: The entries to process.

    :return: The retrieved data.
    """
    tasks = list()
    sem = Semaphore(THREAD_COUNT)
    async with ClientSession() as sess:
        for entry in entries:
            tasks.append(
                wait_for(
                    retrieve_content(entry, sess, sem),
                    timeout=TIMEOUT,
                )
            )
        return await gather(*tasks)


async def download(url, sess, sem):
    """
    Downloads an image from the specified URL.

    :param url: The URL to download the image from.
    :param sess: The session to use to retrieve the data.
    :param sem: Used to limit concurrency.

    :return: The file's data.
    """
    async with sem, sess.get(url) as res:
        logger.info('Downloading %s' % url)
        content = await res.read()
        logger.info('Finished downloading %s' % url)
    if res.status != 200:
        raise ImageNotDownloadedException('Could not download image.')
    return content
```

### Image Prediction

The last two methods tie all of the above code together.

`predict_images` predicts the ImageNet categories for a list of `Pillow` formatted images and returns the results as a list of `ImageOutput` 
- **Security**: Sensitive information, such as Docker Hub credentials, is securely managed using AWS SSM Parameter Store.

## Getting Started

To get started with the GrowthString Project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/growthstring.git
   cd growthstring
2. **Set Up Your Environment**:

Ensure you have Python and Flask installed.
Install Docker and configure it on your machine.
Set up AWS CLI and configure your credentials.

3.**Build and Run the Application**:

    Build the Docker image:
    docker build -t growthstring .
    
    Run the Docker container:
    docker run -p 5000:5000 growthstring
    
4.**Access the Application**:
Open your web browser and navigate to http://localhost:5000.


