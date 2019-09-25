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
python3 run_ace.py --source_dir SOURCE_DIR --test_dir TEST_DIR --working_dir SAVE_DIR --model_to_run InceptionV3 --bottlenecks mixed_8 --num_test 20 --num_random_exp 40 --max_imgs 50 --min_imgs 30
```

where:
```
SOURCE_DIR: Directory where the discovery images (refer to the paper) are saved. 
It should contain two folders: 
1-"target_class" which contains images of the class to be explained. 
2-"random_discovery" which contains randomly selected images of the same dataset.
```

```
TEST_DIR: Used for the profile classifier experiment (not part of the paper).
If None, the profile classifier experiment is not performed.
Same as source_dir:
1-"target_class" which contains test images of the class to be explained. 
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

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
# ACE

