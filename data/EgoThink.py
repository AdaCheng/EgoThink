# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os
from PIL import Image as Img

import datasets


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{Cheng_2024_CVPR,
    author    = {Cheng, Sijie and Guo, Zhicheng and Wu, Jingwen and Fang, Kechen and Li, Peng and Liu, Huaping and Liu, Yang},
    title     = {EgoThink: Evaluating First-Person Perspective Thinking Capability of Vision-Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {14291-14302}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
EgoThink benchmark for VLMs from a first-person perspective.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://adacheng.github.io/EgoThink/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "Activity": "./Activity/",
    "Forecasting": "./Forecast/",
    "Localization_location": "./Localization/location/",
    "Localization_spatial": "./Localization/spatial/",
    "Object_affordance": "./Object/affordance/",
    "Object_attribute": "./Object/attribute/",
    "Object_existence": "./Object/existence/",
    "Planning_assistance": "./Planning/assistance/",
    "Planning_navigation": "./Planning/navigation/",
    "Reasoning_comparing": "./Reasoning/comparing/",
    "Reasoning_counting": "./Reasoning/counting/",
    "Reasoning_situated": "./Reasoning/situated/"
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class EgoThink(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="Activity", version=VERSION, description="This part of my dataset covers a first domain"),
        datasets.BuilderConfig(name="Forecasting", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Localization_location", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Localization_spatial", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Object_affordance", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Object_attribute", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Object_existence", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Planning_assistance", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Planning_navigation", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Reasoning_comparing", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Reasoning_counting", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="Reasoning_situated", version=VERSION, description="This part of my dataset covers a second domain"),
    ]

    DEFAULT_CONFIG_NAME = "Activity"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        # if self.config.name == "first_domain":  # This is the name of the configuration selected in BUILDER_CONFIGS above
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string")
                # These are the features of your dataset like images, labels ...
            }
        )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option2": datasets.Value("string"),
        #             "second_domain_answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)
        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TRAIN,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "train.jsonl"),
            #         "split": "train",
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "dev.jsonl"),
            #         "split": "dev",
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": urls,
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        json_file = filepath + 'annotations.json'
        image_dir = filepath + 'images/'
        with open(json_file, encoding="utf-8") as f:
            file = json.load(f)
            for key, data in enumerate(file):
                # if self.config.name == "first_domain":
                    # Yields examples as (key, example) tuples
                yield key, {
                    "image": Img.open(image_dir + data["image_path"][0]).convert('RGB'),
                    "question": data["question"],
                    "answer": data["answer"],
                }
                # else:
                #     yield key, {
                #         "sentence": data["sentence"],
                #         "option2": data["option2"],
                #         "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                #     }