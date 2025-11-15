# TSIAA

## Teacher-Student Instance-level Adversarial Augmentation for Single Domain Generalized Medical Image Segmentation.(https://ieeexplore.ieee.org/document/11146907)
Wang, Zhengshan and Chen, Long and Xie, Xuelin and Zhang, Yang and Cai, Yunpeng and Ding, Weiping \

**Abstract**

Recently, single-source domain generalization (SDG) has gained popularity in medical image segmentation. As a prominent technique, adversarial image augmentation technique can generate synthetic training data that are challenging for the segmentation model to recognize. To avoid the over-augmentation problem, existing adversarial-based works often employ augmenters with relatively simple structures for medical images, typically operating at the image level, limiting the diversity of the augmented images. In this paper, we propose a Teacher-Student Instance-level Adversarial Augmentation (TSIAA) model for generalized medical image segmentation. The objective of TSIAA is to derive domain-generalizable representations by exploring out-of-source data distributions. First, we construct an Instance-level Image Augmenter (IIAG) using several Instance-level Augmentation Modules (IAMs), which are based on the learnable constrained Bézier transformation function. Compared to image-level adversarial augmentation, instance-level adversarial augmentation breaks the uniformity of augmentation rules across different structures within an image, thereby providing greater diversity. Then, TSIAA conducts Teacher-Student (TS) learning through an adversarial approach, alternating novel image augmentation and generalized representation learning. The former delves into out-of-source and plausible data, while the latter continuously updates both the student and teacher to ensure the original and augmented features maintain consistent and generalized characteristics. By integrating both strategies, our proposed TSIAA model achieves significant improvements over state-of-the-art methods in four challenging SDG tasks. The code can be accessed at https://github.com/Wangzs0228/TSIAA.


## Acknowledgements

Our codes are built upon [SLAug](https://github.com/Kaiseem/SLAug), thanks for their contribution to the community and the development of researches!

## Citation
If our work or code helps you, please consider to cite our paper. Thank you!

```
@ARTICLE{11146907,
  author={Wang, Zhengshan and Chen, Long and Xie, Xuelin and Zhang, Yang and Cai, Yunpeng and Ding, Weiping},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Teacher-Student Instance-level Adversarial Augmentation for Single Domain Generalized Medical Image Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Biomedical imaging;Image segmentation;Data models;Training;Image augmentation;Security;Data augmentation;Adaptation models;Training data;Shape;Single domain generalization;medical image segmentation;image augmentation;adversarial training},
  doi={10.1109/TMI.2025.3605162}}

```
