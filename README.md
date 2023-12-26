# [Code] CITE: Connecting Image and Text Embeddings

<!-- select Model and/or Data and/or Code as needed -->

<!--
**Here are some ideas to get you started:**
🙋‍♀️ A short introduction - what is your organization all about?
🌈 Contribution guidelines - how can the community get involved?
👩‍💻 Useful resources - where can the community find your docs? Is there anything else the community should know?
🍿 Fun facts - what does your team eat for breakfast?
🧙 Remember, you can do mighty things with the power of [Markdown](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
-->


<!-- Insert the project banner here 
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="assets/teaser.png"></a>
</div>
-->

---

<!-- Select some of the point info, feel free to delete -->
<!--
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
[![PyPI](https://img.shields.io/pypi/v/DI-engine)](https://pypi.org/project/DI-engine/)
![Conda](https://anaconda.org/opendilab/di-engine/badges/version.svg)
![Conda update](https://anaconda.org/opendilab/di-engine/badges/latest_release_date.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/DI-engine)
![PyTorch Version](https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com/PaParaZz1/54c5c44eeb94734e276b2ed5770eba8d/raw/85b94a54933a9369f8843cc2cea3546152a75661/badges.json)


![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/comments.json)

![Style](https://github.com/opendilab/DI-engine/actions/workflows/style.yml/badge.svg)
![Docs](https://github.com/opendilab/DI-engine/actions/workflows/doc.yml/badge.svg)
![Unittest](https://github.com/opendilab/DI-engine/actions/workflows/unit_test.yml/badge.svg)
![Algotest](https://github.com/opendilab/DI-engine/actions/workflows/algo_test.yml/badge.svg)
![deploy](https://github.com/opendilab/DI-engine/actions/workflows/deploy.yml/badge.svg)
[![codecov](https://codecov.io/gh/opendilab/DI-engine/branch/main/graph/badge.svg?token=B0Q15JI301)](https://codecov.io/gh/opendilab/DI-engine)

![GitHub Org's stars](https://img.shields.io/github/stars/opendilab)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/DI-engine)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/blob/master/LICENSE)
-->

Updated on 2023.12.26



## Key Features

This repository provides the official implementation of *Text-guided Foundation Model Adaptation for Pathological Image Classification*.

- Foundation model adaptation to medical imaging analysis
- Data-efficient and low-cost visual prompt tuning
- Injection of medical in-domain knowledge via text
- Compatibility with various foundation models


## Links

- [Paper](https://arxiv.org/abs/2307.14901)
- [Code](https://github.com/Yunkun-Zhang/CITE)
- Pre-trained models
  - [CLIP](https://github.com/openai/CLIP)
  - [BioLinkBERT](https://huggingface.co/michiyasunaga/BioLinkBERT-large)
  <!-- [Code] may link to your project at your institute -->


<!-- give a introduction of your project -->
## Details

The recent surge of foundation models in computer vision and natural language processing opens up perspectives in utilizing multi-modal clinical data to train large models with strong generalizability.
Yet pathological image datasets often lack biomedical text annotation and enrichment.
Guiding data-efficient image diagnosis from the use of biomedical text knowledge becomes a substantial interest.
In this paper, we propose to **C**onnect **I**mage and **T**ext **E**mbeddings (CITE) to enhance pathological image classification.
CITE injects text insights gained from language models pre-trained with a broad range of biomedical texts, leading to adapt foundation models towards pathological image understanding.
Through extensive experiments on the PatchGastric stomach tumor pathological image dataset, we demonstrate that  CITE achieves leading performance compared with various baselines especially when training data is scarce. CITE offers insights into leveraging in-domain text knowledge to reinforce data-efficient pathological image classification.

An overview of CITE:
<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="assets/method.png"></a>
</div>



## Dataset

The PatchGastric dataset includes histopathological image patches extracted from H&E stained whole slide images (WSI) of stomach adenocarcinoma endoscopic biopsy specimens. The dataset contains 9 subtypes of gastric adenocarcinoma WSIs. We choose 3 major subtypes including “well differentiated tubular adenocarcinoma”, “moderately differentiated tubular adenocarcinoma”, and “poorly differentiated adenocarcinoma” to form a 3-class grading-like classification task with 179,285 patches of size 300x300 from 693 WSIs.

To prepare the PatchGastric dataset:

1. Download `captions.csv` and `patches_captions.zip` from [PatchGastricADC22](https://zenodo.org/record/6550925).
2. Put them in `data/` and unzip the file.

## Get Started

**Main Requirements**  
> torch==1.13.0  
> mmcls==0.25.0  
> transformers  
> clip   


**Installation**
```bash
conda create -n CITE python=3.9
conda activate CITE
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install openmim
mim install mmcls==0.25.0
pip install -r requirements.txt
```

**Preprocess**

To follow our split of the dataset, please generate the annotation files by running:
```bash
python tools/ann.py
```

Or you can generate your own split following mmcls format:
```text
filename label
```

**Training**

The config files follow [mmcls](https://github.com/open-mmlab/mmclassification) style.
```bash
PYTHONPATH=.:$PYTHONPATH mim train mmcls <config>
```

**Testing**

```bash
PYTHONPATH=.:$PYTHONPATH mim test mmcls <config> --checkpoint <checkpoint> --metrics <metrics>
```

## 🙋‍♀️ Feedback and Contact

- [Yunkun Zhang (Dequan Wang's Group)](mailto:yunkunzhang@dqwang.group)
- [Dequan Wang's Group](https://dqwang.group/)

## 📝 Citation

```text
@inproceedings{zhang2023text,
  title={Text-guided Foundation Model Adaptation for Pathological Image Classification},
  author={Zhang, Yunkun and Gao, Jin and Zhou, Mu and Wang, Xiaosong and Qiao, Yu and Zhang, Shaoting and Wang, Dequan},
  booktitle={MICCAI},
  year={2023}
}
```

## 🗃️ Materials

We provide a comprehensive overview of current open-source medical language models, vision foundation models, and vision-language models, illustrating their applicability to our approach (CITE). For BERT-based language models, you may directly replace `model->head->text_encoder->model` and `model->neck->out_features` with your preferred Huggingface🤗 model in the config file to run CITE.

### Medical Language Models

| Model            | Subfield    | Paper                                                        | Code                                                         | Base    |
| :--------------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :------ |
| Meditron         | Medicine    | [Meditron-70B: Scaling Medical Pretraining for Large Language Models](https://arxiv.org/abs/2311.16079) | [Github](https://github.com/epfLLM/meditron)                 | LLaMA 2 |
| RadFM            | Radiology   | [Towards Generalist Foundation Model for Radiology](https://arxiv.org/abs/2308.02463) | [Github](https://chaoyi-wu.github.io/RadFM)                  | LLaMA   |
| BioMedGPT        | Biomedicine | [BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine](https://arxiv.org/abs/2308.09442) | [Github](https://github.com/PharMolix/OpenBioMed)            | LLaMA 2 |
| Med-PaLM 2       | Clinic      | [Towards Expert-Level Medical Question Answering with Large Language Models](https://arxiv.org/abs/2305.09617) | [Google](https://sites.research.google/med-palm/)            | PaLM 2  |
| PMC-LLaMA        | Medicine    | [PMC-LLaMA: Towards Building Open-source Language Models for Medicine](https://arxiv.org/abs/2304.14454) | [Github](https://github.com/chaoyi-wu/PMC-LLaMA)             | LLaMA   |
| BenTsao (HuaTuo) | Biomedicine | [HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://arxiv.org/abs/2304.06975) | [Github](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) | LLaMA   |
| ChatDoctor       | Medicine    | [ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge](https://arxiv.org/abs/2303.14070) | [Github](https://github.com/Kent0n-Li/ChatDoctor)            | LLaMA   |
| Clinical-T5      | Clinic      | [Clinical-T5: Large Language Models Built Using Mimic Clinical Text](https://www.physionet.org/content/clinical-t5/1.0.0/) | [PhysioNet](https://www.physionet.org/content/clinical-t5/1.0.0/) | T5      |
| Med-PaLM         | Clinic      | [Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138) | [Google](https://sites.research.google/med-palm)             | PaLM    |
| BioGPT           | Biomedicine | [BioGPT: Generative Pre-Trained Transformer for Biomedical Text Generation and Mining](https://academic.oup.com/bib/article-abstract/23/6/bbac409/6713511) | [Github](https://github.com/microsoft/BioGPT)                | GPT-2   |
| BioLinkBERT      | Biomedicine | [Linkbert: Pretraining Language Models with Document Links](https://arxiv.org/abs/2203.15827) | [Github](https://github.com/michiyasunaga/LinkBERT)          | BERT    |
| PubMedBERT       | Biomedicine | [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://arxiv.org/abs/2007.15779) | [Microsoft](https://microsoft.github.io/BLURB/models.html)   | BERT    |
| BioBERT          | Biomedicine | [BioBERT: A Pre-Trained Biomedical Language Representation Model for Biomedical Text Mining](https://academic.oup.com/bioinformatics/article-abstract/36/4/1234/5566506) | [Github](https://github.com/naver/biobert-pretrained)        | BERT    |
| BlueBERT         | Biomedicine | [An Empirical Study of Multi-Task Learning on BERT for Biomedical Text Mining](https://arxiv.org/abs/2005.02799) | [Github](https://github.com/ncbi-nlp/BLUE_Benchmark)         | BERT    |
| Clinical BERT    | Clinic      | [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323) | [Github](https://github.com/EmilyAlsentzer/clinicalBERT)     | BERT    |
| SciBERT          | Biomedicine | [SciBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676) | [Github](https://github.com/allenai/scibert)                 | BERT    |

### Vision Models

| Model          | Subfield    | Paper                                                        | Code                                                         | Base   |
| :------------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----- |
| REMEDIS        | Radiology   | [Robust and Data-Efficient Generalization of Self-Supervised Machine Learning for Diagnostic Imaging](https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/s41551-023-01049-7&casa_token=jsWqfcJssI0AAAAA:zt3n5PYal2WyePCxeKXW4q4x0gmqtWQYHCLqXbLQhK1ERML3pgp68Q7GBN1wVK9MYP5iyxBzlsaD1Tygag) | [Github](https://github.com/google-research/medical-ai-research-foundations) | SimCLR |
| RETFound       | Retinopathy | [A Foundation Model for Generalizable Disease Detection from Retinal Images](https://www.nature.com/articles/s41586-023-06555-x) | [Github](https://github.com/rmaphoh/RETFound_MAE)            | MAE    |
| CTransPath     | Pathology   | [Transformer-Based Unsupervised Contrastive Learning for Histopathological Image Classification](https://www.sciencedirect.com/science/article/pii/S1361841522002043?casa_token=YBbUxnv_qsAAAAAA:YrgecQ6ecLad4Bj3JfGl0SZvjRgSQBZ27KYtpH6jU3vy6j-8hGrnQzbVFWCg0vH9Pn7r5H1Cxw) | [Github](https://github.com/Xiyue-Wang/TransPath)            | -      |
| HIPT           | Pathology   | [Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Scaling_Vision_Transformers_to_Gigapixel_Images_via_Hierarchical_Self-Supervised_Learning_CVPR_2022_paper.html?trk=public_post_comment-text) | [Github](https://github.com/mahmoodlab/HIPT)                 | DINO   |
| INTERN-2.5     | General     | [InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions](https://arxiv.org/abs/2211.05778) | [Github](https://github.com/OpenGVLab/InternImage)           | -      |
| DINOv2         | General     | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | [Github](https://github.com/facebookresearch/dinov2)         | -      |
| MAE            | General     | [Masked Autoencoders are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper) | [Github](https://github.com/facebookresearch/mae)            | -      |
| ViT (ImageNet) | General     | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | [Huggingface](https://huggingface.co/google/vit-base-patch16-224-in21k) | -      |

### Vision-Language Models

| Model        | Subfield    | Paper                                                        | Code                                                         | Base     |
| :----------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :------- |
| Qilin-Med-VL | Radiology   | [Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare](https://arxiv.org/abs/2310.17956) | [Github](https://github.com/williamliujl/Qilin-Med-VL)       | LLaVA    |
| RadFM        | Radiology   | [Towards Generalist Foundation Model for Radiology](https://arxiv.org/abs/2308.02463) | [Github](https://github.com/chaoyi-wu/RadFM)                 | -        |
| KAD          | Radiology   | [Knowledge-Enhanced Visual-Language Pre-Training on Chest Radiology Images](https://www.nature.com/articles/s41467-023-40260-7) | [Github](https://github.com/xiaoman-zhang/KAD)               | CLIP     |
| Med-Flamingo | Medicine    | [Med-Flamingo: A Multimodal Medical Few-Shot Learner](https://proceedings.mlr.press/v225/moor23a.html) | [Github](https://github.com/snap-stanford/med-flamingo)      | Flamingo |
| QuiltNet     | Pathology   | [Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/abs/2306.11207) | [Github](https://github.com/wisdomikezogwo/quilt1m)          | CLIP     |
| PLIP         | Pathology   | [A Visual-Language Foundation Model for Pathology Image Analysis Using Medical Twitter](https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/s41591-023-02504-3&casa_token=cnEpAWMo9RIAAAAA:_v3_yKPcr_afGn_MCirdOLLHyC63vSFVuvqu2sM4lnxJaZVQF7gmZsEjP2-W-CTQ9Xr2OVOpQEjgdIf9Jw) | [Huggingface](https://huggingface.co/spaces/vinid/webplip)   | CLIP     |
| MI-Zero      | Pathology   | [Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images](http://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html) | [Github](https://github.com/mahmoodlab/MI-Zero)              | CLIP     |
| LLaVA-Med    | Biomedicine | [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890) | [Github](https://github.com/microsoft/LLaVA-Med)             | LLaVA    |
| MedVInT      | Biomedicine | [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://arxiv.org/abs/2305.10415) | [Github](https://github.com/xiaoman-zhang/PMC-VQA)           | -        |
| PMC-CLIP     | Biomedicine | [PMC-CLIP: Contrastive Language-Image Pre-Training Using Biomedical Documents](https://arxiv.org/abs/2303.07240) | [Github](https://github.com/WeixiongLin/PMC-CLIP)            | CLIP     |
| BiomedCLIP   | Biomedicine | [Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing](https://arxiv.org/abs/2303.00915) | [Huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | CLIP     |
| MedCLIP      | Medicine    | [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://arxiv.org/abs/2210.10163) | [Github](https://github.com/RyanWangZf/MedCLIP)              | CLIP     |
| CheXzero     | Radiology   | [Expert-Level Detection of Pathologies from Unannotated Chest X-ray Images via Self-Supervised Learning](https://www.nature.com/articles/s41551-022-00936-9) | [Github](https://github.com/rajpurkarlab/CheXzero)           | CLIP     |
| PubMedCLIP   | Radiology   | [Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?](https://arxiv.org/abs/2112.13906) | [Github](https://github.com/sarahESL/PubMedCLIP)             | CLIP     |
| LLaVA        | Genearl     | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | [Github](https://github.com/haotian-liu/LLaVA)               | -        |
| Flamingo     | General     | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) | -        |
| CLIP         | General     | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | [Github](https://github.com/openai/CLIP)                     | -        |

