
# MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix
[**📖 arXiv**](https://arxiv.org/abs/2505.13032) | [**🎬 MMAR Demo Video**](https://www.youtube.com/watch?v=Dab13opIGqU) | [**🛠️ GitHub Code**](https://github.com/ddlBoJack/MMAR) | [**🔊 MMAR Audio Download (HuggingFace)**](https://huggingface.co/datasets/BoJack/MMAR)
                                          
<p align="center"><img src="assets/logo.png" alt="MMAR Benchmark Logo" width="300"/></p>

## Overview of MMAR
We introduce MMAR, a new benchmark designed to evaluate the deep reasoning capabilities of Audio-Language Models (ALMs) across massive multi-disciplinary tasks. 
MMAR comprises 1,000 meticulously curated audio-question-answer triplets, collected from real-world internet videos and refined through iterative error corrections and quality checks to ensure high quality. 
Each item in the benchmark demands multi-step deep reasoning beyond surface-level understanding. Moreover, a part of the questions requires graduate-level perceptual and domain-specific knowledge, elevating the benchmark's difficulty and depth. 
Examples include:

![Example](assets/example.png)

The metadata for MMAR is available in [this file](MMAR-meta.json). Unlike previous benchmarks, MMAR not only covers traditional modalities such as speech, audio, and music, but also extends to their mix, collected from in-the-wild videos. The distribution of data across these modalities is illustrated in the left figure. Furthermore, each question is annotated with a designated category and sub-category, as shown in the right figure.

For each question, we also provide the URL and corresponding timestamp of the original video, as well as the spoken language (if present) in the clip. To prevent potential data leakage into training for reasoning models, we have withheld reasoning cues and chain-of-thought annotations, which will be released at an appropriate time.

<p float="left">
  <img src="assets/modality_pie.png" width="49%" />
  <img src="assets/category_sunburst.png" width="49%" />
</p>

## Leaderboard
We benchmark the models (open-source or API) on MMAR across five model categories: 
1. Large Audio Language Models (LALMs)
2. Large Audio Reasoning Models (LARMs)
3. Omni Language Models (OLMs)
4. Large Language Models (LLMs) with audio captions as input
5. Large Reasoning Models (LRMs) with audio captions as input

| Models                                  | Size  | Avg    | Sound   | Music   | Speech  | Sound-Music | Sound-Speech | Music-Speech | Sound-Music-Speech |
|-----------------------------------------|-------|--------|---------|---------|---------|-------------|--------------|--------------|--------------------|
| Random Guess                            | -     | 29.32  | 29.39   | 25.88   | 31.48   | 25.00       | 29.30        | 31.10        | 28.13              |
|-----------------------------------------|-------|--------|---------|---------|---------|-------------|--------------|--------------|--------------------|
| **Large Audio Language Models (LALMs)** |       |        |         |         |         |             |              |              |                    |
| 🏅GPT-4o Audio                            | -     | 63.50  | 53.94   | 50.97   | 70.41   | 63.64       | 72.48        | 62.20        | 75.00              |
| 🥈GPT-4o mini Audio                       | -     | 50.60  | 38.79   | 35.92   | 58.84   | 45.45       | 60.09        | 57.32        | 50.00              |
| 🥉R1-AQA                                  | 8.4B  | 47.60  | 55.76   | 37.38   | 48.98   | 9.09        | 50.00        | 50.00        | 50.00              |
| SALMONN                                 | 7B    | 32.80  | 30.91   | 29.61   | 34.35   | 9.09        | 37.61        | 28.05        | 37.50              |
| Qwen2-Audio                             | 8.4B  | 30.40  | 33.94   | 23.30   | 32.99   | 9.09        | 33.03        | 26.83        | 33.33              |
| Qwen2-Audio-Instruct                    | 8.4B  | 30.00  | 33.33   | 24.27   | 32.31   | 9.09        | 31.19        | 30.49        | 25.00              |
| Audio Flamingo                          | 2.2B  | 26.60  | 32.73   | 21.84   | 24.83   | 18.18       | 30.28        | 24.39        | 25.00              |
| GAMA                                    | 7B    | 26.50  | 29.09   | 24.27   | 27.89   | 27.27       | 24.77        | 28.05        | 20.83              |
| Qwen-Audio-Chat                         | 8.4B  | 23.50  | 27.88   | 20.39   | 22.11   | 9.09        | 25.23        | 25.61        | 20.83              |
| Audio Flamingo 2                        | 0.5B  | 23.00  | 20.61   | 20.39   | 24.15   | 27.27       | 23.85        | 26.83        | 25.00              |
| Audio Flamingo 2                        | 1.5B  | 22.90  | 26.67   | 20.87   | 22.79   | 9.09        | 22.94        | 23.17        | 20.83              |
| Audio Flamingo 2                        | 3B    | 21.90  | 24.85   | 17.48   | 20.75   | 18.18       | 26.61        | 23.17        | 8.33               |
| LTU                                     | 7B    | 19.20  | 19.39   | 19.90   | 13.95   | 18.18       | 24.77        | 21.95        | 16.67              |
| LTU-AS                                  | 7B    | 19.00  | 20.00   | 14.08   | 19.05   | 9.09        | 20.64        | 28.05        | 12.50              |
| GAMA-IT                                 | 7B    | 17.40  | 22.42   | 16.02   | 12.24   | 36.36       | 22.48        | 14.63        | 12.50              |
| MU-LLaMA                                | 7B    | 13.90  | 13.94   | 13.59   | 14.97   | 9.09        | 12.39        | 14.63        | 16.67              |
| MusiLingo                               | 7B    | 6.60   | 9.09    | 7.28    | 4.08    | 9.09        | 6.88         | 7.32         | 8.33               |
|-----------------------------------------|-------|--------|---------|---------|---------|-------------|--------------|--------------|--------------------|
| **Large Audio Reasoning Models (LARMs)**|       |        |         |         |         |             |              |              |                    |
| 🏅Audio-Reasoner                          | 8.4B  | 36.80  | 43.64   | 33.50   | 32.99   | 45.45       | 42.66        | 31.71        | 25.00              |
| 🥈Audio-CoT                               | 8.4B  | 31.30  | 35.76   | 25.24   | 34.01   | 9.09        | 30.73        | 30.49        | 37.50              |
| 🥉Mellow                                  | 167M  | 30.00  | 33.33   | 26.70   | 24.83   | 18.18       | 37.16        | 32.93        | 29.17              |
|-----------------------------------------|-------|--------|---------|---------|---------|-------------|--------------|--------------|--------------------|
| **Omni Language Models (OLMs)**         |       |        |         |         |         |             |              |              |                    |
| 🏅Gemini 2.0 Flash                        | -     | 65.60  | 61.21   | 50.97   | 72.11   | 81.82       | 72.48        | 65.85        | 70.83              |
| 🥈Qwen-2.5-Omni                           | 7B    | 56.70  | 58.79   | 40.78   | 59.86   | 54.55       | 61.93        | 67.07        | 58.33              |
| 🥉Qwen-2.5-Omni                           | 3B    | 53.80  | 53.94   | 46.12   | 53.74   | 36.36       | 60.09        | 57.32        | 58.33              |
| Baichuan-Omni-1.5                       | 11B   | 40.70  | 41.21   | 33.01   | 40.48   | 36.36       | 48.62        | 39.02        | 41.67              |
| OpenOmni                                | 8B    | 27.00  | 20.61   | 22.33   | 35.37   | 18.18       | 27.06        | 23.17        | 25.00              |
| AnyGPT-chat                             | 8B    | 23.70  | 24.24   | 19.42   | 22.11   | 27.27       | 27.52        | 26.83        | 29.17              |
|-----------------------------------------|-------|--------|---------|---------|---------|-------------|--------------|--------------|--------------------|
| **Large Language Models (LLMs)**        |       |        |         |         |         |             |              |              |                    |
| Caption + GPT-4o                        | -     | 50.70  | 46.06   | 40.29   | 60.88   | 27.27       | 53.67        | 46.34        | 45.83              |
| Caption + DeepSeek-V3                   | 671B  | 47.60  | 42.42   | 40.78   | 56.12   | 18.18       | 50.00        | 45.12        | 37.50              |
|-----------------------------------------|-------|--------|---------|---------|---------|-------------|--------------|--------------|--------------------|
| **Large Reasoning Models (LRMs)**       |       |        |         |         |         |             |              |              |                    |
| Caption + DeepSeek-R1                   | 671B  | 55.50  | 46.67   | 49.51   | 62.59   | 45.45       | 58.72        | 56.10        | 54.17              |
| Caption + OpenAI o3                     | -     | 54.70  | 49.70   | 41.75   | 63.95   | 36.36       | 60.09        | 52.44        | 54.17              |
| Caption + OpenAI o1                     | -     | 53.00  | 48.48   | 43.20   | 63.61   | 18.18       | 56.88        | 45.12        | 45.83              |



## Dataset Creation
The MMAR benchmark was constructed with a comprehensive pipeline. The process includes: 
1. Brainstorming challenging questions
2. Building a taxonomy through human-LLM collaboration
3. Heuristic-based data collection and annotation
4. Crawling audio data and enriching content across multiple slots
5. Performing iterative correction and quality inspection to ensure high data fidelity

![Pipeline](assets/pipeline.png)

## Test Your Model !

To ensure a smooth integration into existing evaluation pipelines, we adopt an evaluation methodology modified from [MMAU](https://github.com/Sakshi113/MMAU), implemented in [evaluation.py](code/evaluation.py). The input to the evaluation script should be the same as [MMAR-meta.json](MMAR-meta.json), with an additional key named `model_prediction`, which stores the model prediction for each question. 
  
To run the script:
```bash
python evaluation.py  --input INPUT_JSON_PATH
```

## Acknowledge
We gratefully acknowledge that our evaluation code is modified from the official implementation of [MMAU](https://github.com/Sakshi113/MMAU). 

## Citation
```
@article{ma2025mmar,
  title={MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix},
  author={Ma, Ziyang and Ma, Yinghao and Zhu, Yanqiao and Yang, Chen and Chao, Yi-Wen and Xu, Ruiyang and others},
  journal={Proc. NeurIPS},
  year={2025}
}
```
