# Project Title

ACE: Towards Automatic Concept Based Explanations

## Getting Started
Here is the tensorflow implementations of the paper [Towards Automatic Concept-based Explanations](https://arxiv.org/abs/1902.03129) presented at NeurIPS 2019.

### Prerequisites

Required python libraries:

```
  Scikit-image: https://scikit-image.org/
  Tensorflow: https://www.tensorflow.org/
  TCAV: https://github.com/tensorflow/tcav
```

### Installing

An example run command:

```
python3 run_ace.py --num_parallel_runs 0 --target_class Zebra --source_dir SOURCE_DIR ---working_dir SAVE_DIR --model_to_run InceptionV3 --model_path PATH_TO_MODEL_CHECKPOINT --bottlenecks mixed_8 --num_test 20 --num_random_exp 40 --max_imgs 50 --min_imgs 30 --test_dir TEST_DIR 
```

where:
```
num_parallel_runs: Number of parallel jobs (loading images, etc). If 0, parallel processing is deactivated.
```

```
target_class: Name of the class to be explained.
```

```
SOURCE_DIR: Directory where the discovery images (refer to the paper) are saved. 
It should contain (at least) two folders: 
1-"target_class" which contains images of the class to be explained. 
2-"random_discovery" which contains randomly selected images of the same dataset.
```

```
SAVE_DIR: Where the experiment results (both text report and the discovered concept examples) are saved.
```

```
model_to_run: One of InceptionV3 or GoogleNet is supported. You can change the "make_model" function in ace_helpers.py to have your own customized model.
model_path: Path to the model's saved graph.
```

```
TEST_DIR: Used for the profile classifier experiment (not part of the paper).
If None, the profile classifier experiment is not performed.
Same as source_dir:
1-"Name of the target class (here zebra)" which contains test images of the class to be explained. 
2-"random_test" which contains test images randomly selected from the test data.
```
End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Amirata Ghorbani** - [Website](http://web.stanford.edu/~amiratag)
* **James Wexler** - [Website](https://ai.google/research/people/105507/)
* **James Zou** - [Website](https://sites.google.com/site/jamesyzou/)
* **Been Kim** - [Website](https://beenkim.github.io/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Work was done as part of Google Brain internship.
# ACE

