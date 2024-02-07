
<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset">
    <img src="images/logo.png" alt="Logo" width="240" height="240">
  </a>

  <h3 align="center">Temporal Logic Video (TLV) Dataset</h3>

  <p align="center">
    Synthetic and real video dataset with temporal logic annotation
    <br />
    <a href="https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset">View Demo</a>
    ·
    <a href="https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset/issues">Report Bug</a>
    ·
    <a href="https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

Given the lack of SOTA video datasets for long-horizon,
temporally extended activity and object detection, we intro-
duce the Temporal Logic Video (TLV) datasets. The syn-
thetic TLV datasets are compiled by stitching together static
images from computer vision datasets like COCO and
ImageNet. This enables the artificial introduction of
a wide range of TL specifications. Additionally, we have
created two video datasets based on the open-source au-
tonomous vehicle (AV) driving datasets NuScenes and
Waymo.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

If you want to generate syntetic dataset from COCO and ImageNet, you should download the source data first. 

1. [ImageNet](https://image-net.org/challenges/LSVRC/2017/index.php): The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2017. Recommended file structure as follows: 
```
|--ILSVRC
|----Annotations
|----Data
|----ImageSets
|----LOC_synset_mapping.txt
```

2. [COCO](https://cocodataset.org/#download): Download the source data as follow:
```
|--COCO
|----2017
|------annotations
|------train2017
|------val2017
```

### Installation
```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build
python -m pip install --editable ."[dev, test]"
```
   
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
Please find argument details from run scripts. 

### Data Loader Common Argument
* `data_root_dir`: The root directory where the COCO dataset is stored.
* `mapping_to`: Map the original label to desired mapper, default is "coco".
* `save_dir`: Directory where the generated dataset will be saved.
### Synthetic Generator Common Argument
* `initial_number_of_frame`: Initial number of frames for each video.
* `max_number_frame`: Maximum number of frames for each video.
* `number_video_per_set_of_frame`: Number of videos to generate per set of frames.
* `increase_rate`: Rate at which the number of frames increases.
* `ltl_logic`: Temporal logic to apply. Options include various logical expressions like "F prop1", "G prop1", etc.
* `save_frames`: Boolean to decide whether to save individual frames (True or False).
In each run script, make sure 

1. **coco synthetic data generator** <br>
COCO synthetic data generator can generate & compositions since it has multiple labels.
```
python3 run_scripts/run_synthetic_tlv_coco.py --data_root_dir "../COCO/2017" --save_dir "<output dir path>"
```

2. **Imagenet synthetic data generator** <br>
Imagenet synthetic data generator cannot generate & LTL logic formula.
```
python3 run_synthetic_tlv_imagenet.py --data_root_dir "../ILSVRC" --save_dir "<output dir path>""
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Publication
   - [ ] Repository
   - [ ] Blog

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING 
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Minkyu Choi - [@your_twitter](https://twitter.com/MinkyuChoi7) - minkyu.choi@utexas.edu

Project Link: TBD

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* University of Texas at Austin (UT Austin)
* UT Austin Swarm Lab

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/UTAustin-SwarmLab/temporal-logic-video-dataset/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/mchoi07/
[product-screenshot]: images/screenshot.png
