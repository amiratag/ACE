# ACE

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
python3 run_ace.py --num_parallel_runs 0 --target_class zebra --source_dir SOURCE_DIR ---working_dir SAVE_DIR --model_to_run GoogleNet --model_path ./tensorflow_inception_graph.pb --labels_path ./imagenet_labels.txt -bottlenecks mixed4c --num_random_exp 40 --max_imgs 50 --min_imgs 30
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
model_to_run: One of InceptionV3 or GoogleNet is supported (the weights are provided for GoogleNet). You can change the "make_model" function in ace_helpers.py to have your own customized model.
model_path: Path to the model's saved graph.
```


## Authors

* **Amirata Ghorbani** - [Website](http://web.stanford.edu/~amiratag)
* **James Wexler** - [Website](https://ai.google/research/people/105507/)
* **James Zou** - [Website](https://sites.google.com/site/jamesyzou/)
* **Been Kim** - [Website](https://beenkim.github.io/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Work was done as part of Google Brain internship.

