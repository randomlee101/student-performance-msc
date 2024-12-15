"""StudentPerformance Dataset"""

from typing import List
from functools import partial

import datasets

import pandas

VERSION = datasets.Version("1.0.0")
_BASE_FEATURE_NAMES = [
    "sex",
    "ethnicity",
    "parental_level_of_education",
    "has_standard_lunch",
    "has_completed_preparation_test",
    "math_score",
    "reading_score",
    "writing_score"
]

_ENCODING_DICS = {
    "sex": {
        "female": 0,
        "male": 1
    },
    "parental_level_of_education": {
        "some high school": 0,
        "high school": 1,
        "some college": 2,
        "bachelor's degree": 3,
        "master's degree": 4,
        "associate's degree": 5,
    },
    "has_standard_lunch": {
        "free/reduced": 0,
        "standard": 1
    },
    "has_completed_preparation_test": {
        "none": 0,
        "completed": 1
    }
}

DESCRIPTION = "StudentPerformance dataset."
_HOMEPAGE = "https://www.kaggle.com/datasets/ulrikthygepedersen/student_performances"
_URLS = ("https://www.kaggle.com/datasets/ulrikthygepedersen/student_performances")
_CITATION = """"""

# Dataset info
urls_per_split = {
    "train": "https://huggingface.co/datasets/mstz/student_performance/raw/main/student_performance.csv",
}
features_types_per_config = {
    "encoding": {
        "feature": datasets.Value("string"),
        "original_value": datasets.Value("string"),
        "encoded_value": datasets.Value("int64")
    },
    "math": {
        "is_male": datasets.Value("bool"),
        "ethnicity": datasets.Value("string"),
        "parental_level_of_education": datasets.Value("int8"),
        "has_standard_lunch": datasets.Value("bool"),
        "has_completed_preparation_test": datasets.Value("bool"),
        "reading_score": datasets.Value("int64"),
        "writing_score": datasets.Value("int64"),
        "has_passed_math_exam": datasets.ClassLabel(num_classes=2, names=("no", "yes"))
    },
    "writing": {
        "is_male": datasets.Value("bool"),
        "ethnicity": datasets.Value("string"),
        "parental_level_of_education": datasets.Value("int8"),
        "has_standard_lunch": datasets.Value("bool"),
        "has_completed_preparation_test": datasets.Value("bool"),
        "reading_score": datasets.Value("int64"),
        "math_score": datasets.Value("int64"),
        "has_passed_writing_exam": datasets.ClassLabel(num_classes=2, names=("no", "yes")),
    },
    "reading": {
        "is_male": datasets.Value("bool"),
        "ethnicity": datasets.Value("string"),
        "parental_level_of_education": datasets.Value("int8"),
        "has_standard_lunch": datasets.Value("bool"),
        "has_completed_preparation_test": datasets.Value("bool"),
        "writing_score": datasets.Value("int64"),
        "math_score": datasets.Value("int64"),
        "has_passed_reading_exam": datasets.ClassLabel(num_classes=2, names=("no", "yes")),
    }
}
features_per_config = {k: datasets.Features(features_types_per_config[k]) for k in features_types_per_config}


class StudentPerformanceConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(StudentPerformanceConfig, self).__init__(version=VERSION, **kwargs)
        self.features = features_per_config[kwargs["name"]]


class StudentPerformance(datasets.GeneratorBasedBuilder):
    # dataset versions
    DEFAULT_CONFIG = "math"
    BUILDER_CONFIGS = [
        StudentPerformanceConfig(name="encoding",
                                 description="Encoding dictionaries."),
        StudentPerformanceConfig(name="math",
                                 description="Binary classification, predict if the student has passed the math exam."),
        StudentPerformanceConfig(name="reading",
                                 description="Binary classification, predict if the student has passed the reading exam."),
        StudentPerformanceConfig(name="writing",
                                 description="Binary classification, predict if the student has passed the writing exam."),
    ]

    def _info(self):
        if self.config.name not in features_per_config:
            raise ValueError(f"Unknown configuration: {self.config.name}")

        info = datasets.DatasetInfo(description=DESCRIPTION, citation=_CITATION, homepage=_HOMEPAGE,
                                    features=features_per_config[self.config.name])

        return info

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloads = dl_manager.download_and_extract(urls_per_split)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloads["train"]}),
        ]

    def _generate_examples(self, filepath: str):
        if self.config.name not in features_types_per_config:
            raise ValueError(f"Unknown config: {self.config.name}")
        elif self.config.name == "encoding":
            data = self.encoding_dics()
        else:
            data = pandas.read_csv(filepath)
            data = self.preprocess(data, config=self.config.name)

        for row_id, row in data.iterrows():
            data_row = dict(row)

            yield row_id, data_row

    def preprocess(self, data: pandas.DataFrame, config: str = "math") -> pandas.DataFrame:
        data.columns = _BASE_FEATURE_NAMES
        for feature in _ENCODING_DICS:
            encoding_function = partial(self.encode, feature)
            data.loc[:, feature] = data[feature].apply(encoding_function)
        data = data.rename(columns={"sex": "is_male"})
        data = data.astype({"is_male": "bool", "has_standard_lunch": "bool", "has_completed_preparation_test": "bool"})

        if config == "math":
            data = data.rename(columns={"math_score": "has_passed_math_exam"})
            data.loc[:, "has_passed_math_exam"] = data.has_passed_math_exam.apply(lambda x: int(x > 60))

            return data[list(features_types_per_config["math"].keys())]
        elif config == "reading":
            data = data.rename(columns={"reading_score": "has_passed_reading_exam"})
            data.loc[:, "has_passed_reading_exam"] = data.has_passed_reading_exam.apply(lambda x: int(x > 60))

            return data[list(features_types_per_config["reading"].keys())]
        elif config == "writing":
            data = data.rename(columns={"writing_score": "has_passed_writing_exam"})
            data.loc[:, "has_passed_writing_exam"] = data.has_passed_writing_exam.apply(lambda x: int(x > 60))

            return data[list(features_types_per_config["writing"].keys())]

    def encode(self, feature, value):
        return _ENCODING_DICS[feature][value]

    def encoding_dics(self):
        data = [pandas.DataFrame([(feature, original, encoded) for original, encoded in d.items()])
                for feature, d in _ENCODING_DICS.items()]
        data = pandas.concat(data, axis="rows").reset_index()
        data.drop("index", axis="columns", inplace=True)
        data.columns = ["feature", "original_value", "encoded_value"]

        return data

