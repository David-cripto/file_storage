import torch

import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import tqdm

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning
old_merge_environment_settings = requests.Session.merge_environment_settings


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


class CMMDDataset(Dataset):
    def __init__(self, imgs_ar, reshape_to, max_count=-1):
        self.imgs_ar = imgs_ar
        # self.resize_func = v2.Resize(reshape_to, interpolation=InterpolationMode.BICUBIC)
        self.max_count = max_count

    def __len__(self):
        return len(self.imgs_ar)

    def __getitem__(self, idx):
        # return self.resize_func(self.imgs_ar[idx])
        return self.imgs_ar[idx]


def compute_embeddings_for_dir(
    imgs_ar,
    embedding_model,
    batch_size,
    max_count=-1,
):
    print(f"embedding_model.input_image_size = {embedding_model.input_image_size}")
    dataset = CMMDDataset(imgs_ar, reshape_to=embedding_model.input_image_size, max_count=max_count)
    count = len(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_embs = []
    for batch in tqdm.tqdm(dataloader, total=count // batch_size):
        image_batch = batch.numpy()

        # Normalize to the [0, 1] range.
        # image_batch = image_batch / 255.0
        # image_batch = (image_batch + 1)/2
        if np.min(image_batch) < 0 or np.max(image_batch) > 1:
            raise ValueError(
                "Image values are expected to be in [0, 1]. Found:" f" [{np.min(image_batch)}, {np.max(image_batch)}]."
            )
        image_batch = image_batch.transpose(0, 2, 3, 1)

        # Compute the embeddings using a pmapped function.
        embs = np.asarray(
            embedding_model.embed(image_batch)
        )  # The output has shape (num_devices, batch_size, embedding_dim).
        all_embs.append(embs)

    all_embs = np.concatenate(all_embs, axis=0)

    return all_embs


def mmd(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    # The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
    # details.
    _SIGMA = 10
    # The following is used to make the metric more human readable. See the paper
    # for more details.
    _SCALE = 1000

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)


def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images


class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        _CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
        self._CUDA_AVAILABLE = torch.cuda.is_available()

        with no_ssl_verification():
            self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)
            self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        if self._CUDA_AVAILABLE:
            self._model = self._model.cuda()

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images):
        """Computes CLIP embeddings for the given images.

        Args:
          images: An image array of shape (batch_size, height, width, 3). Values are
            in range [0, 1].

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """

        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if self._CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs


def compute_cmmd(eval_imgs_ar, ref_embed_file, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    embedding_model = ClipEmbeddingModel()
    ref_embs = np.load(ref_embed_file).astype("float32")
    eval_embs = compute_embeddings_for_dir(eval_imgs_ar, embedding_model, batch_size, max_count).astype("float32")
    val = mmd(ref_embs, eval_embs)
    # compute_cmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value)

    return val.numpy()

def save_ref(ref_ar, ref_embed_file, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    embedding_model = ClipEmbeddingModel()
    ref_embs = compute_embeddings_for_dir(ref_ar, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    np.save(ref_embed_file, ref_embs)