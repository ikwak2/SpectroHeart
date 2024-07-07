# Participation of [Heart murmur detection challenge (2022 PhysioNet Challenge)](https://moody-challenge.physionet.org/)
<!--
This repository contains the source for a manuscript to appear in ...

- Datasets for the code can be downloaded from the [challenge website](https://moody-challenge.physionet.org/2022/)

---
-->

This repository contains the code from our team's participation in the 2022 heart murmur detection competition, as well as the code related to the extended work published in our subsequent paper, "SpectroHeart: A Deep Neural Network Approach to Heart Murmur Detection Using Spectrogram and Peak Interval Features." 

## Data Access  
The training data of the George B. Moody PhysioNet Challenge 2022 can be downloaded from [PhysioNet](https://physionet.org/content/circor-heart-sound/1.0.3/) [1]. You can also download it directly using this [link](https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip) or the following command:
```
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
```

## Description of the files
Here's a description of the files:

- 소율학생 실험 코드들 ipynb 형식으로 몇개 올려주세요~ 설명도 여기 달아주세요~



## Useful links
* [The George B. Moody PhysioNet Challenge 2022 website](https://moody-challenge.physionet.org/2022/)
* [Baseline code from the challenge](https://github.com/physionetchallenges/python-classifier-2022)
* [Evaluation code from the challenge](https://github.com/physionetchallenges/evaluation-2022)
* [Results for the 2022 challenge](https://moody-challenge.physionet.org/2022/results/)
* [The link](https://physionet.org/static/published-projects/challenge-2022/1.0.0/sources/) provides source code from the teams that participated in the competition.
* [PeakUtils package code](https://github.com/lucashn/peakutils?tab=readme-ov-file)

## References
[1] Heart Murmur Detection from Phonocardiogram Recordings: The George B.
Moody PhysioNet Challenge 2022
```
@inproceedings{reyna2022heart,
  title={Heart murmur detection from phonocardiogram recordings: The george b. moody physionet challenge 2022},
  author={Reyna, Matthew A and Kiarashi, Yashar and Elola, Andoni and Oliveira, Jorge and Renna, Francesco and Gu, Annie and Alday, Erick A Perez and Sadr, Nadi and Sharma, Ashish and Mattos, Sandra and others},
  booktitle={2022 Computing in Cardiology (CinC)},
  volume={498},
  pages={1--4},
  year={2022},
  organization={IEEE}
}
```

[2] Application of LCNN model to PCG data.
```
@inproceedings{lee2022deep,
  title={Deep Learning Based Heart Murmur Detection Using Frequency-time Domain Features of Heartbeat Sounds},
  author={Lee, Jungguk and Kang, Taein and Kim, Narin and Han, Soyul and Won, Hyejin and Gong, Wuming and Kwak, Il-Youp},
  booktitle={2022 Computing in Cardiology (CinC)},
  volume={498},
  pages={1--4},
  year={2022},
  organization={IEEE}
}
```
