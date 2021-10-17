<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<!--<p align="center">
  <a href="https://github.com/habibcomsats/monaifl">
    <img src="images/logo.png" alt="Logo" width="120" height="80">
  </a>
-->
  <h3 align="center">MONAI-FL</h3>

  <p align="center">
    This repository contains the basic federated learning example of monai code considering substra's opener and algo classes.
    <!--<br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
    -->
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
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
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<p align="left">
MONAI-FL is planned to be a tool-agnostic federated learning platform for medical imaging applications. We envision the below-mentioned architecture for our code-base, however, please feel free to modify the code-base considering your own application requirements.
</p>
<p align="center">
<img src="images/monaifl.png" alt="MONAI-FL" align="center">
</p>
<!--
Here's a blank template to get started:
**To avoid retyping too much info. Do a search and replace with your text editor for the following:**
`github_username`, `repo_name`, `twitter_handle`, `email`, `project_title`, `project_description`
-->

### Built With

* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [Google gRPC](https://grpc.io/)
* [Project MONAI](https://monai.io/)
* [Pytorch](https://pytorch.org/get-started/locally/)
* [Conda](https://docs.conda.io/en/latest/)
* [Ubuntu](https://ubuntu.com/)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Following platforms are neccessary to run this project.
* Linux Installation (preferably Ubuntu distributions). [For other Operating Systems, you can try and share your experience with us :-)]   

* install visual studio code
  ```sh
  https://code.visualstudio.com/docs/setup/linux
  ```
* install conda environment
  ```sh
  https://docs.conda.io/en/latest/
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/habibcomsats/monaifl.git
   ```
2. Create conda environment
   ```sh
   conda create -n monaifl python=3
   ```
3. Activate conda environment
   ```sh
   conda activate monaifl
   ```
4. Install your system compatible pytorch version
   ```sh
   https://pytorch.org/get-started/locally/
   ```
5. Install MONAI with Nibabel and TQDM
  ```sh
  https://docs.monai.io/en/latest/installation.html
  ```
6. Install grpc with python compiler
  ```sh
  python -m pip install grpcio
  ```
  ```sh
  python -m pip install grpcio-tools
  ```
7. Install Pandas
  ```sh
  conda install pandas
  ```
9. Install Numpy
  ```sh
  conda install numpy
  ```
<!-- USAGE EXAMPLES -->
## Running Centralized FL Workflows

</p>
<p align="center">
<img src="images/Hub-n-Spoke.png" alt="MONAI-FL" align="center">
</p>

Reference notebook
https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb

Mount the dataset:
 This repository contains the demo dataset. If you are cloning this repository, you do not need to mount any dataset. Otherwise, you must execute the cell 5 from the reference notebook, extract the dataset at your preferred location and set the path in testmonai.py 

Running FL Nodes first:
1. Open a new terminal
2. Activate conda environment
3. Navigate into code repository (currently it is not dockerized)
  ```sh
  $ cd monaifl/hubnspoke
  ```
4. Run the node at each terminal@($monaifl/hubnspoke/)
  ```sh
    python flnode/node1.py
    python flnode/node2.py
  ```
Running Hub:
1. Open a new terminal
2. Activate conda environment
3. Navigate into code repository (currently it is not dockerized)
  ```sh
  $ cd monaifl/hubnspoke
  ```
4. Run the hub@($monaifl/hubnspoke/)
  ```sh
  python hub/start.py
  ```
CAUTION: The Hub side code does not require any GPU installation but for the client side you must have a CUDA-enabled device to initiate the training. 

## Running Decentralized FL Workflows
Reference notebook

https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb

Mount the dataset:
 This repository contains the demo dataset. If you are cloning this repository, you do not need to mount any dataset. Otherwise, you must execute the cell 5 from the reference notebook, extract the dataset at your preferred location and set the path in testmonai.py 

Running Server:
1. Open a new terminal
2. Activate conda environment
3. Navigate into code repository (currently it is not dockerized)
  ```sh
  $ cd monaifl/decentral_fl
  ```
4. Run the server@($monaifl/decentral_fl/)
  ```sh
    python aggregator/src/server.py
  ```
Running Client:
1. Open two new terminals
2. Activate conda environment in both terminals
3. Navigate into code repository (currently it is not dockerized)
  ```sh
  $ cd monaifl/decentral_fl
  ```
4. Run the client at each terminal@($monaifl/decentral_fl/)
  ```sh
  python trainer/client1.py
  python trainer/client2.py
  ```
CAUTION: The server side code does not require any GPU installation but for the client side you must have a CUDA-enabled device to initiate the training. 

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/habibcomsats/monaifl/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Habib Rehman - [@habibcomsats](https://twitter.com/habibcomsats) - email (habibcomsats@gmail.com or mhrehman@ieee.org)

Project Link: [https://github.com/habibcomsats/monaifl](https://github.com/habibcomsats/monaifl)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [AMIGO](https://github.com/AmigoLab)
* [AnswerDigital](https://answerdigital.com/)
* [AI Center for Value-Based Healthcare](https://www.aicentre.co.uk/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!--[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
-->
