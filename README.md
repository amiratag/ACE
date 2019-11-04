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
python3 ace_run.py --num_parallel_runs 0 --target_class zebra --source_dir SOURCE_DIR --working_dir SAVE_DIR --model_to_run GoogleNet --model_path ./tensorflow_inception_graph.pb --labels_path ./imagenet_labels.txt --bottlenecks mixed4c --num_random_exp 40 --max_imgs 50 --min_imgs 30
```

where:
```
num_random_exp: number of random concepts with respect to which concept-activaion-vectors are computed for calculating the TCAV score of a discovered concept (recommended >20).
```
For example if you set num_random_exp=20, you need to create folders random500_0, rando500_1, ..., random_500_19 and put them in the SOURCE_DIR where each folder contains a set of 50-500 randomly selected images of the dataset (ImageNet in our case). 

```
target_class: Name of the class which prediction is to be explained.
```

```
SOURCE_DIR: Directory where the discovery images (refer to the paper) are saved. 
It should contain (at least) num_random_exp + 2 folders: 
1-"target_class" which contains images of the class to be explained (in this example the shoulder should be names as zebra). 
2-"random_discovery" which contains randomly selected images of the same dataset (at lease $max_imgs number of images).
3-"random500_0, ..., random_500_${num_random_exp} where each one contains 500 randomly selected images from the data set"
```

```
num_parallel_runs: Number of parallel jobs (loading images, etc). If 0, parallel processing is deactivated.
```


```
SAVE_DIR: Where the experiment results (both text report and the discovered concept examples) are saved.
```

```
model_to_run: One of InceptionV3 or GoogleNet is supported (the weights are provided for GoogleNet). You can change the "make_model" function in ace_helpers.py to have your own customized model.
model_path: Path to the model's saved graph.
```
If you are using a custom model, you should write a wrapper for it containing the following methods:
```
run_examples(images, BOTTLENECK_LAYER): which basically returens the activations of the images in the BOTTLENECK_LAYER. 'images' are original images without preprocessing (float between 0 and 1)
get_image_shape(): returns the shape of the model's input
label_to_id(CLASS_NAME): returns the id of the given class name.
get_gradient(activations, CLASS_ID, BOTTLENECK_LAYER): computes the gradient of the CLASS_ID logit in the logit layer with respect to activations in the BOTTLENECK_LAYER.
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

