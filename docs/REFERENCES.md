# References

Publication status is included where it materially affects evidence weight. “Preprint/technical report” does not imply poor work; it means claims may not have completed archival peer review.

## StreamVC and voice conversion

1. Yang et al. “StreamVC: Real-Time Low-Latency Voice Conversion.” ICASSP 2024. [Paper](https://arxiv.org/abs/2401.03078), [DOI](https://doi.org/10.1109/ICASSP48485.2024.10446863), [project](https://google-research.github.io/seanet/stream_vc/).
2. van Niekerk et al. “A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion.” ICASSP 2022. [Paper](https://arxiv.org/abs/2111.02392).
3. Chen et al. “Streaming Voice Conversion via Intermediate Bottleneck Features and Non-Streaming Teacher Guidance.” ICASSP 2023.
4. Jia et al. “FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion.” ICASSP 2023. [Paper](https://arxiv.org/abs/2210.15418).
5. Guo et al. “QuickVC: Any-to-many Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion.” Preprint, 2023. [Paper](https://arxiv.org/abs/2302.08296).
6. Liu et al. “Zero-shot Voice Conversion with Diffusion Transformers” (Seed-VC). Preprint, 2024. [Paper](https://arxiv.org/abs/2411.09943), [repository](https://github.com/Plachtaa/seed-vc).
7. Qin et al. “OpenVoice: Versatile Instant Voice Cloning.” Preprint, 2023. [Paper](https://arxiv.org/abs/2312.01479), [repository](https://github.com/myshell-ai/OpenVoice).
8. Wang et al. “StreamVoice: Streamable Context-Aware Language Modeling for Real-time Zero-Shot Voice Conversion.” ACL 2024. [Paper](https://arxiv.org/abs/2401.11053).
9. “Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion.” Preprint, 2025. [Paper](https://arxiv.org/abs/2507.14534).
10. “Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization.” Preprint, June 2026. [Paper](https://arxiv.org/abs/2606.20218).
11. Popov et al. “Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme.” ICLR 2022. [Paper](https://arxiv.org/abs/2109.13821).

## Neural codecs and tokenizers

12. Zeghidour et al. “SoundStream: An End-to-End Neural Audio Codec.” IEEE/ACM TASLP. [Paper](https://arxiv.org/abs/2107.03312).
13. Défossez et al. “High Fidelity Neural Audio Compression” (EnCodec). Preprint, 2022. [Paper](https://arxiv.org/abs/2210.13438), [repository](https://github.com/facebookresearch/encodec).
14. Kumar et al. “High-Fidelity Audio Compression with Improved RVQGAN” (DAC). NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.06546), [repository](https://github.com/descriptinc/descript-audio-codec).
15. Défossez et al. “Moshi: a Speech-Text Foundation Model for Real-Time Dialogue” (includes Mimi). Technical report/preprint, 2024. [Paper](https://arxiv.org/abs/2410.00037), [repository](https://github.com/kyutai-labs/moshi).
16. Siuzdak et al. “SNAC: Multi-Scale Neural Audio Codec.” NeurIPS 2024 Audio Imagination Workshop/preprint. [Paper](https://arxiv.org/abs/2410.14411), [OpenReview](https://openreview.net/forum?id=PFBF5ctj4X).
17. Zhang et al. “SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models.” ICLR 2024. [Paper](https://arxiv.org/abs/2308.16692), [repository](https://github.com/ZhangXInFD/SpeechTokenizer).
18. Ye et al. “Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model” (X-Codec). AAAI 2025. [Paper](https://arxiv.org/abs/2408.17175), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34761).
19. Ye et al. “LLaSA: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis” (X-Codec2). Preprint, 2025. [Paper](https://arxiv.org/abs/2502.04128).
20. Mentzer et al. “Finite Scalar Quantization: VQ-VAE Made Simple.” ICLR 2024. [Paper](https://arxiv.org/abs/2309.15505), [OpenReview](https://openreview.net/forum?id=8ishA3LxN8).
21. Borsos et al. “AudioLM: a Language Modeling Approach to Audio Generation.” IEEE/ACM TASLP. [Paper](https://arxiv.org/abs/2209.03143).
22. “WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling.” Preprint, 2024. [Paper](https://arxiv.org/abs/2408.16532).
23. “BigCodec: Pushing the Limits of Low-Bitrate Neural Speech Codec.” Preprint, 2024. [Paper](https://arxiv.org/abs/2409.05377).

## Streaming and efficient sequence models

24. Gulati et al. “Conformer: Convolution-augmented Transformer for Speech Recognition.” Interspeech 2020. [Paper](https://arxiv.org/abs/2005.08100).
25. Rekesh et al. “Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition.” ASRU 2023. [Paper](https://arxiv.org/abs/2305.05084).
26. Shi et al. “Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition.” ICASSP 2021. [Paper](https://arxiv.org/abs/2010.10759).
27. “Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition.” Preprint, 2023. [Paper](https://arxiv.org/abs/2312.17279).
28. “Dynamic Chunk Convolution for Unified Streaming and Non-Streaming Conformer ASR.” Preprint, 2023. [Paper](https://arxiv.org/abs/2304.09325).
29. Gu and Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” Preprint, 2023; subsequent ICML lineage. [Paper](https://arxiv.org/abs/2312.00752).
30. Fang and Li. “Mamba for Streaming ASR Combined with Unimodal Aggregation.” ICASSP 2025. [Paper](https://arxiv.org/abs/2410.00070).
31. “Speech-Mamba: Long-Context Speech Recognition with Selective State Spaces Models.” Preprint, 2024. [Paper](https://arxiv.org/abs/2409.18654).
32. Poli et al. “Hyena Hierarchy: Towards Larger Convolutional Language Models.” ICML 2023. [Paper](https://arxiv.org/abs/2302.10866).
33. Dao. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.” ICLR 2024. [Paper](https://arxiv.org/abs/2307.08691).

## Speech representations and speaker models

34. Hsu et al. “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.” IEEE/ACM TASLP, 2021. [Paper](https://arxiv.org/abs/2106.07447).
35. Chen et al. “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.” IEEE JSTSP, 2022. [Paper](https://arxiv.org/abs/2110.13900).
36. Babu et al. “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale.” Interspeech 2022. [Paper](https://arxiv.org/abs/2111.09296).
37. Radford et al. “Robust Speech Recognition via Large-Scale Weak Supervision” (Whisper). ICML 2023. [Paper](https://arxiv.org/abs/2212.04356).
38. Chen et al. “EAT: Self-Supervised Pre-Training with Efficient Audio Transformer.” IJCAI 2024. [Paper](https://arxiv.org/abs/2401.03497).
39. Desplanques et al. “ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.” Interspeech 2020. [Paper](https://arxiv.org/abs/2005.07143).
40. Ju et al. “NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models” (FACodec). ICML 2024. [Paper](https://arxiv.org/abs/2403.03100).

## TTS and speech foundation models

41. Du et al. “CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens.” Preprint, 2024. [Paper](https://arxiv.org/abs/2407.05407).
42. Du et al. “CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models.” Preprint, 2024. [Paper](https://arxiv.org/abs/2412.10117), [repository](https://github.com/FunAudioLLM/CosyVoice).
43. “Fun-CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training.” Preprint, 2025. [Paper](https://arxiv.org/abs/2505.17589).
44. Wang et al. “MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer.” ICLR 2025. [Paper](https://arxiv.org/abs/2409.00750).
45. Chen et al. “F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.” ACL 2025. [Paper](https://arxiv.org/abs/2410.06885), [repository](https://github.com/SWivid/F5-TTS).
46. “FireRedTTS-1S: An Upgraded Streamable Foundation Text-to-Speech System.” Preprint, 2025. [Paper](https://arxiv.org/abs/2503.20499), [repository](https://github.com/FireRedTeam/FireRedTTS).
47. “Fish Audio S2 Technical Report.” Preprint/vendor report, 2026. [Paper](https://arxiv.org/abs/2603.08823), [repository](https://github.com/fishaudio/fish-speech).
48. Sesame AI. “Crossing the Uncanny Valley of Conversational Voice” (CSM). Technical blog/model release, 2025. [Article](https://www.sesame.com/blog/crossing-the-uncanny-valley-of-voice), [repository](https://github.com/SesameAILabs/csm).
49. Zeng et al. “GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot.” Preprint, 2024. [Paper](https://arxiv.org/abs/2412.02612), [repository](https://github.com/zai-org/GLM-4-Voice).
50. Labiausse et al. “High-Fidelity Simultaneous Speech-To-Speech Translation” (Hibiki). ICML 2025. [Paper](https://arxiv.org/abs/2502.03382), [repository](https://github.com/kyutai-labs/hibiki).

## Vocoders and synthesis

51. Kong et al. “HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.” NeurIPS 2020. [Paper](https://arxiv.org/abs/2010.05646).
52. Lee et al. “BigVGAN: A Universal Neural Vocoder with Large-Scale Training.” ICLR 2023. [Paper](https://arxiv.org/abs/2206.04658).
53. Kim et al. “Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech” (VITS). ICML 2021. [Paper](https://arxiv.org/abs/2106.06103).

## Quantization and deployment

54. Lin et al. “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.” MLSys 2024. [Paper](https://arxiv.org/abs/2306.00978).
55. Frantar et al. “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.” ICLR 2023. [Paper](https://arxiv.org/abs/2210.17323).
56. Xiao et al. “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.” ICML 2023. [Paper](https://arxiv.org/abs/2211.10438).
57. ONNX Runtime. [Graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), [memory tuning](https://onnxruntime.ai/docs/performance/tune-performance/memory.html).
58. NVIDIA TensorRT. [Dynamic shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html).
59. Apple Core ML Tools. [Stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html).
60. Google AI Edge. [LiteRT documentation](https://developers.google.com/edge/litert).
61. PyTorch. [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/).
62. “Pushing the Limits of On-Device Streaming ASR: A Compact High-Accuracy English Model for Low-Latency Inference.” Preprint/Microsoft Research, 2026. [Paper](https://arxiv.org/abs/2604.14493).

## Evaluation, data, and safety

63. Zen et al. “LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech.” Interspeech 2019. [Paper](https://arxiv.org/abs/1904.02882).
64. Ardila et al. “Common Voice: A Massively-Multilingual Speech Corpus.” LREC 2020. [Paper](https://arxiv.org/abs/1912.06670).
65. Reddy et al. “DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric.” ICASSP 2021.
66. VoicePrivacy Challenge. [VoicePrivacy 2024](https://www.voiceprivacychallenge.org/).
67. European Union. [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj).

